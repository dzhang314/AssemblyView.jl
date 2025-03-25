module AssemblyView


####################################################### EXTRACTING ASSEMBLY CODE


using InteractiveUtils: code_native


function assembly_lines(@nospecialize(f), @nospecialize(types))
    buffer = IOBuffer()
    code_native(buffer, f, types; syntax=:intel, debuginfo=:default,
        dump_module=true, binary=false, raw=false)
    return split(String(take!(buffer)), '\n'; keepempty=false)
end


###################################################### PARSING METADATA COMMENTS


using Base.Iterators: partition


const BLOCK_OPEN_REGEX = r"^; (│*)┌ @ (.*) within `(.*)`(?: @ (.*))?$"
const BLOCK_CONTINUE_REGEX = r"^; (│*) @ (.*) within `(.*)`(?: @ (.*))?$"
const BLOCK_CLOSE_REGEX = r"^; (│*)(└+)$"


struct SourceLocation
    file_name::SubString{String}
    line_number::Union{Nothing,Int}
end


function SourceLocation(s::SubString{String})
    colon_index = findlast(':', s)
    if isnothing(colon_index)
        return SourceLocation(s, nothing)
    else
        try
            file_name = @view s[begin:colon_index-1]
            line_string = @view s[colon_index+1:end]
            line_number = parse(Int, line_string; base=10)
            return SourceLocation(file_name, line_number)
        catch
            SourceLocation(s, nothing)
        end
    end
end


struct SourceContext
    origin::SourceLocation
    function_name::SubString{String}
    overload_path::Vector{SourceLocation}
end


# The AssemblyInstruction datatype is architecture-agnostic and makes
# no assumptions about the syntax of a particular assembly language.
# It simply acts as a container for the metadata printed by `code_native`.
struct AssemblyInstruction
    code::SubString{String}
    context::Vector{SourceContext}
end


struct AssemblyLabel
    name::SubString{String}
end


function is_label(line::SubString{String})
    if startswith(line, '\t')
        return false
    end
    hash_index = findfirst('#', line)
    if !isnothing(hash_index)
        line = rstrip(@view line[begin:hash_index-1])
    end
    return endswith(line, ':')
end


function extract_label(line::SubString{String})
    @assert !startswith(line, '\t')
    hash_index = findfirst('#', line)
    if !isnothing(hash_index)
        line = rstrip(@view line[begin:hash_index-1])
    end
    @assert endswith(line, ':')
    return AssemblyLabel(@view line[begin:end-1])
end


function parse_metadata(lines::Vector{SubString{String}})
    context_stack = SourceContext[]
    result = Union{AssemblyInstruction,AssemblyLabel}[]
    for line in lines

        if startswith(line, "\t.")
            recognized =
                (line == "\t.text") ||
                startswith(line, "\t.file\t") ||
                startswith(line, "\t.size\t") ||
                startswith(line, "\t.type\t") ||
                startswith(line, "\t.global\t") ||
                startswith(line, "\t.globl\t") ||
                startswith(line, "\t.p2align\t") ||
                startswith(line, "\t.section\t")
            if !recognized
                @warn "Ignoring unrecognized assembler directive: $line"
            end

        elseif is_label(line)
            push!(result, extract_label(line))

        elseif startswith(lstrip(line), '#')
            # TODO: Decide which comments should be kept or discarded.

        elseif startswith(line, ';')
            # The output of `code_native` contains `debuginfo` comments that
            # specify the location of the Julia source code corresponding to
            # each assembly instruction. We use regular expressions to parse
            # this information.
            block_open_match = match(BLOCK_OPEN_REGEX, line)
            block_continue_match = match(BLOCK_CONTINUE_REGEX, line)
            block_close_match = match(BLOCK_CLOSE_REGEX, line)
            @assert +(
                !isnothing(block_open_match),
                !isnothing(block_continue_match),
                !isnothing(block_close_match),
            ) <= 1 # At most one match can occur.
            if !isnothing(block_open_match)
                # A block_open comment indicates that all subsequent assembly
                # instructions are generated from a particular Julia function
                # until a corresponding block_continue or block_close comment
                # is reached. Blocks may nest to indicate multiple levels of
                # Julia functions being inlined into the current function.
                stack_str, loc_str, func_name, path_str = block_open_match
                @assert all(c == '│' for c in stack_str)
                @assert length(context_stack) == length(stack_str)
                if isnothing(path_str)
                    push!(context_stack, SourceContext(
                        SourceLocation(loc_str), func_name, SourceLocation[]
                    ))
                else
                    push!(context_stack, SourceContext(
                        SourceLocation(loc_str), func_name,
                        SourceLocation.(split(path_str, " @ "))
                    ))
                end
            elseif !isnothing(block_continue_match)
                # A block_continue comment closes one block and immediately
                # opens another block in the same line.
                stack_str, loc_str, func_name, path_str = block_continue_match
                @assert all(c == '│' for c in stack_str)
                @assert length(context_stack) == length(stack_str)
                pop!(context_stack)
                if isnothing(path_str)
                    push!(context_stack, SourceContext(
                        SourceLocation(loc_str), func_name, SourceLocation[]
                    ))
                else
                    push!(context_stack, SourceContext(
                        SourceLocation(loc_str), func_name,
                        SourceLocation.(split(path_str, " @ "))
                    ))
                end
            elseif !isnothing(block_close_match)
                # A block_close comment closes one or more open blocks,
                # indicated by the number of '└' characters.
                stack_str, close_str = block_close_match
                @assert all(c == '│' for c in stack_str)
                @assert all(c == '└' for c in close_str)
                for _ in close_str
                    pop!(context_stack)
                end
                @assert length(context_stack) == length(stack_str)
            else
                @warn "Ignoring unrecognized debuginfo comment:\n$line"
            end

        elseif startswith(line, '\t')
            push!(result, AssemblyInstruction(line, copy(context_stack)))

        else
            @warn "Ignoring unrecognized assembly code:\n$line"
        end
    end
    @assert isempty(context_stack)
    return result
end


########################################################### PARSING X86 ASSEMBLY


const X86_REGISTERS = Dict{String,Tuple{Symbol,Int}}([
    "al" => (:R0, 8), "ah" => (:R0, 8), "ax" => (:R0, 16),
    "eax" => (:R0, 32), "rax" => (:R0, 64),
    "cl" => (:R1, 8), "ch" => (:R1, 8), "cx" => (:R1, 16),
    "ecx" => (:R1, 32), "rcx" => (:R1, 64),
    "dl" => (:R2, 8), "dh" => (:R2, 8), "dx" => (:R2, 16),
    "edx" => (:R2, 32), "rdx" => (:R2, 64),
    "bl" => (:R3, 8), "bh" => (:R3, 8), "bx" => (:R3, 16),
    "ebx" => (:R3, 32), "rbx" => (:R3, 64),
    "spl" => (:R4, 8), "sp" => (:R4, 16),
    "esp" => (:R4, 32), "rsp" => (:R4, 64),
    "bpl" => (:R5, 8), "bp" => (:R5, 16),
    "ebp" => (:R5, 32), "rbp" => (:R5, 64),
    "sil" => (:R6, 8), "si" => (:R6, 16),
    "esi" => (:R6, 32), "rsi" => (:R6, 64),
    "dil" => (:R7, 8), "di" => (:R7, 16),
    "edi" => (:R7, 32), "rdi" => (:R7, 64),
    "r8b" => (:R8, 8), "r8w" => (:R8, 16),
    "r8d" => (:R8, 32), "r8" => (:R8, 64),
    "r9b" => (:R9, 8), "r9w" => (:R9, 16),
    "r9d" => (:R9, 32), "r9" => (:R9, 64),
    "r10b" => (:R10, 8), "r10w" => (:R10, 16),
    "r10d" => (:R10, 32), "r10" => (:R10, 64),
    "r11b" => (:R11, 8), "r11w" => (:R11, 16),
    "r11d" => (:R11, 32), "r11" => (:R11, 64),
    "r12b" => (:R12, 8), "r12w" => (:R12, 16),
    "r12d" => (:R12, 32), "r12" => (:R12, 64),
    "r13b" => (:R13, 8), "r13w" => (:R13, 16),
    "r13d" => (:R13, 32), "r13" => (:R13, 64),
    "r14b" => (:R14, 8), "r14w" => (:R14, 16),
    "r14d" => (:R14, 32), "r14" => (:R14, 64),
    "r15b" => (:R15, 8), "r15w" => (:R15, 16),
    "r15d" => (:R15, 32), "r15" => (:R15, 64),
    "xmm0" => (:V0, 128), "ymm0" => (:V0, 256), "zmm0" => (:V0, 512),
    "xmm1" => (:V1, 128), "ymm1" => (:V1, 256), "zmm1" => (:V1, 512),
    "xmm2" => (:V2, 128), "ymm2" => (:V2, 256), "zmm2" => (:V2, 512),
    "xmm3" => (:V3, 128), "ymm3" => (:V3, 256), "zmm3" => (:V3, 512),
    "xmm4" => (:V4, 128), "ymm4" => (:V4, 256), "zmm4" => (:V4, 512),
    "xmm5" => (:V5, 128), "ymm5" => (:V5, 256), "zmm5" => (:V5, 512),
    "xmm6" => (:V6, 128), "ymm6" => (:V6, 256), "zmm6" => (:V6, 512),
    "xmm7" => (:V7, 128), "ymm7" => (:V7, 256), "zmm7" => (:V7, 512),
    "xmm8" => (:V8, 128), "ymm8" => (:V8, 256), "zmm8" => (:V8, 512),
    "xmm9" => (:V9, 128), "ymm9" => (:V9, 256), "zmm9" => (:V9, 512),
    "xmm10" => (:V10, 128), "ymm10" => (:V10, 256), "zmm10" => (:V10, 512),
    "xmm11" => (:V11, 128), "ymm11" => (:V11, 256), "zmm11" => (:V11, 512),
    "xmm12" => (:V12, 128), "ymm12" => (:V12, 256), "zmm12" => (:V12, 512),
    "xmm13" => (:V13, 128), "ymm13" => (:V13, 256), "zmm13" => (:V13, 512),
    "xmm14" => (:V14, 128), "ymm14" => (:V14, 256), "zmm14" => (:V14, 512),
    "xmm15" => (:V15, 128), "ymm15" => (:V15, 256), "zmm15" => (:V15, 512),
    "xmm16" => (:V16, 128), "ymm16" => (:V16, 256), "zmm16" => (:V16, 512),
    "xmm17" => (:V17, 128), "ymm17" => (:V17, 256), "zmm17" => (:V17, 512),
    "xmm18" => (:V18, 128), "ymm18" => (:V18, 256), "zmm18" => (:V18, 512),
    "xmm19" => (:V19, 128), "ymm19" => (:V19, 256), "zmm19" => (:V19, 512),
    "xmm20" => (:V20, 128), "ymm20" => (:V20, 256), "zmm20" => (:V20, 512),
    "xmm21" => (:V21, 128), "ymm21" => (:V21, 256), "zmm21" => (:V21, 512),
    "xmm22" => (:V22, 128), "ymm22" => (:V22, 256), "zmm22" => (:V22, 512),
    "xmm23" => (:V23, 128), "ymm23" => (:V23, 256), "zmm23" => (:V23, 512),
    "xmm24" => (:V24, 128), "ymm24" => (:V24, 256), "zmm24" => (:V24, 512),
    "xmm25" => (:V25, 128), "ymm25" => (:V25, 256), "zmm25" => (:V25, 512),
    "xmm26" => (:V26, 128), "ymm26" => (:V26, 256), "zmm26" => (:V26, 512),
    "xmm27" => (:V27, 128), "ymm27" => (:V27, 256), "zmm27" => (:V27, 512),
    "xmm28" => (:V28, 128), "ymm28" => (:V28, 256), "zmm28" => (:V28, 512),
    "xmm29" => (:V29, 128), "ymm29" => (:V29, 256), "zmm29" => (:V29, 512),
    "xmm30" => (:V30, 128), "ymm30" => (:V30, 256), "zmm30" => (:V30, 512),
    "xmm31" => (:V31, 128), "ymm31" => (:V31, 256), "zmm31" => (:V31, 512),
    "k0" => (:K0, 64), "k1" => (:K1, 64), "k2" => (:K2, 64), "k3" => (:K3, 64),
    "k4" => (:K4, 64), "k5" => (:K5, 64), "k6" => (:K6, 64), "k7" => (:K7, 64),
])


const X86_POINTER_SIZES = Dict{String,Int}(
    "byte" => 8, "word" => 16, "dword" => 32, "qword" => 64,
    "tbyte" => 80, "xmmword" => 128, "ymmword" => 256, "zmmword" => 512,
)


const X86_ADDRESS_OPERAND_REGEX = r"^\[(.*)\]$"
const X86_POINTER_OPERAND_REGEX = r"^(.*) ptr \[(.*)\]$"
const X86_OFFSET_OPERAND_REGEX = r"^offset (.*)$"
const X86_LABEL_OPERAND_REGEX = r"^\.LBB0_[0-9]+$"
const X86_INTEGER_OPERAND_REGEX = r"^-?[0-9]+$"


struct X86RegisterOperand
    id::Symbol
    size::Int
    origin::Int
end


function X86RegisterOperand(name::SubString{String})
    @assert haskey(X86_REGISTERS, name)
    id, size = X86_REGISTERS[name]
    if endswith(name, 'h')
        @assert size == 8
        return X86RegisterOperand(id, size, 8)
    else
        return X86RegisterOperand(id, size, 0)
    end
end


# TODO: Parse x86 addressing modes.
struct X86AddressOperand
    expr::SubString{String}
end


struct X86PointerOperand
    address::X86AddressOperand
    size::Int
end


struct X86OffsetOperand
    name::SubString{String}
end


struct X86LabelOperand
    name::SubString{String}
end


struct X86IntegerOperand
    value::Int
end


struct X86SymbolOperand
    name::SubString{String}
end


const X86Operand = Union{
    X86RegisterOperand,
    X86AddressOperand,
    X86PointerOperand,
    X86OffsetOperand,
    X86LabelOperand,
    X86IntegerOperand,
    X86SymbolOperand,
}


function parse_x86_operand(op::SubString{String})
    if haskey(X86_REGISTERS, op)
        return X86RegisterOperand(op)
    else
        address_match = match(X86_ADDRESS_OPERAND_REGEX, op)
        pointer_match = match(X86_POINTER_OPERAND_REGEX, op)
        offset_match = match(X86_OFFSET_OPERAND_REGEX, op)
        label_match = match(X86_LABEL_OPERAND_REGEX, op)
        integer_match = match(X86_INTEGER_OPERAND_REGEX, op)
        @assert +(
            !isnothing(address_match),
            !isnothing(pointer_match),
            !isnothing(offset_match),
            !isnothing(label_match),
            !isnothing(integer_match),
        ) <= 1 # At most one match can occur.
        if !isnothing(address_match)
            return X86AddressOperand(address_match[1])
        elseif !isnothing(pointer_match)
            return X86PointerOperand(
                X86AddressOperand(pointer_match[2]),
                X86_POINTER_SIZES[pointer_match[1]],
            )
        elseif !isnothing(offset_match)
            return X86OffsetOperand(offset_match[1])
        elseif !isnothing(label_match)
            return X86LabelOperand(op)
        elseif !isnothing(integer_match)
            return X86IntegerOperand(parse(Int, op; base=10))
        else
            return X86SymbolOperand(op)
        end
    end
end


struct X86Instruction
    opcode::SubString{String}
    operands::Vector{X86Operand}
    comment::Union{Nothing,SubString{String}}
    context::Vector{SourceContext}
end


function X86Instruction(instruction::AssemblyInstruction)
    code = instruction.code
    @assert startswith(code, '\t')
    hash_index = findfirst('#', code)
    if isnothing(hash_index)
        comment = nothing
        code = strip(code)
    else
        comment = @view code[hash_index:end]
        code = strip(@view code[begin:hash_index-1])
    end
    tab_index = findfirst('\t', code)
    if isnothing(tab_index)
        return X86Instruction(code, X86Operand[], comment, instruction.context)
    else
        opcode = @view code[begin:tab_index-1]
        if opcode == "nop"
            return X86Instruction(
                opcode, X86Operand[], comment, instruction.context)
        end
        operands_string = @view code[tab_index+1:end]
        operands = parse_x86_operand.(strip.(split(operands_string, ',')))
        return X86Instruction(opcode, operands, comment, instruction.context)
    end
end


function Base.print(io::IO, op::X86RegisterOperand)
    printstyled(io, op.id; bold=true)
    printstyled(io, '[', op.origin, ':', op.origin + op.size - 1, ']';
        color=:yellow)
    return nothing
end


Base.print(io::IO, op::X86AddressOperand) =
    printstyled(io, op.expr; bold=true)


function Base.print(io::IO, op::X86PointerOperand)
    printstyled(io, "*("; color=:blue)
    printstyled(io, op.address.expr; color=:blue, bold=true)
    printstyled(io, ")"; color=:blue)
    printstyled(io, "[0:", op.size - 1, ']'; color=:yellow)
    return nothing
end


Base.print(io::IO, op::X86OffsetOperand) =
    printstyled(io, "offset ", op.name; color=:red, bold=true)
Base.print(io::IO, op::X86LabelOperand) =
    printstyled(io, op.name; color=:red, bold=true)
Base.print(io::IO, op::X86IntegerOperand) =
    printstyled(io, op.value; color=:yellow)
Base.print(io::IO, op::X86SymbolOperand) =
    printstyled(io, "<<<", op.name, ">>>"; color=:red, bold=true)


##################################################### ASSEMBLY PARSING INTERFACE


export parsed_asm


function parsed_asm(@nospecialize(f), @nospecialize(types...))
    @static if Sys.ARCH == :x86_64
        lines = parse_metadata(assembly_lines(f, types))
        result = Union{X86Instruction,AssemblyLabel}[]
        for line in lines
            if line isa AssemblyInstruction
                push!(result, X86Instruction(line))
            elseif line isa AssemblyLabel
                push!(result, line)
            else
                @assert false
            end
        end
        return result
    else
        error("Parsing assembly for architecture $(Sys.ARCH) " *
              "is not yet supported by AssemblyView.jl")
    end
end


########################################################### VIEWING X86 ASSEMBLY


const X86_PRINT_HANDLERS =
    Dict{Tuple{String,Vector{DataType}},Function}()


X86_PRINT_HANDLERS[("nop", DataType[])] = ::X86Instruction -> nothing


X86_PRINT_HANDLERS[("ret", DataType[])] = instruction::X86Instruction -> begin
    print('\t')
    printstyled("return"; color=:magenta, bold=true)
    println()
    return nothing
end


X86_PRINT_HANDLERS[(
    "vaddpd",
    [X86RegisterOperand, X86RegisterOperand, X86RegisterOperand]
)] = instruction::X86Instruction -> begin
    a, b, c = instruction.operands
    @assert a.size == b.size == c.size
    println("\t<$(div(a.size, 64)) x f64> $(a.id) .= $(b.id) .+ $(c.id);")
    return nothing
end


X86_PRINT_HANDLERS[(
    "vsubpd",
    [X86RegisterOperand, X86RegisterOperand, X86RegisterOperand]
)] = instruction::X86Instruction -> begin
    a, b, c = instruction.operands
    @assert a.size == b.size == c.size
    println("\t<$(div(a.size, 64)) x f64> $(a.id) .= $(b.id) .- $(c.id);")
    return nothing
end


X86_PRINT_HANDLERS[(
    "vmulpd",
    [X86RegisterOperand, X86RegisterOperand, X86RegisterOperand]
)] = instruction::X86Instruction -> begin
    a, b, c = instruction.operands
    @assert a.size == b.size == c.size
    println("\t<$(div(a.size, 64)) x f64> $(a.id) .= $(b.id) .* $(c.id);")
    return nothing
end


X86_PRINT_HANDLERS[(
    "vdivpd",
    [X86RegisterOperand, X86RegisterOperand, X86RegisterOperand]
)] = instruction::X86Instruction -> begin
    a, b, c = instruction.operands
    @assert a.size == b.size == c.size
    println("\t<$(div(a.size, 64)) x f64> $(a.id) .= $(b.id) ./ $(c.id);")
    return nothing
end


##################################################### ASSEMBLY VIEWING INTERFACE


export view_asm


function view_asm(@nospecialize(f), @nospecialize(types...))
    @static if Sys.ARCH == :x86_64
        for line in parsed_asm(f, types...)
            if line isa X86Instruction
                key = (line.opcode, [typeof(op) for op in line.operands])
                if haskey(X86_PRINT_HANDLERS, key)
                    X86_PRINT_HANDLERS[key](line)
                else
                    print('\t')
                    printstyled(line.opcode; color=:green, bold=true)
                    print(' ')
                    first_op = true
                    for op in line.operands
                        if first_op
                            first_op = false
                        else
                            print(", ")
                        end
                        print(op)
                    end
                    if !isnothing(line.comment)
                        print(' ')
                        printstyled(line.comment; color=:cyan, italic=true)
                    end
                    println()
                end
            elseif line isa AssemblyLabel
                printstyled(line.name; color=:red, bold=true)
                println(':')
            else
                @assert false
            end
        end
    else
        for line in parse_metadata(assembly_lines(f, types))
            if line isa AssemblyInstruction
                println(line.code)
            elseif line isa AssemblyLabel
                printstyled(line.name; color=:red, bold=true)
                println(':')
            else
                @assert false
            end
        end
    end
end


################################################################################

end # module AssemblyView
