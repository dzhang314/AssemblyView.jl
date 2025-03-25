module AssemblyView


####################################################### EXTRACTING ASSEMBLY CODE


using InteractiveUtils: code_native


# TODO: Is dump_module=true or dump_module=false more appropriate here?
# They generate different assembly code, and I'm not sure which is more
# faithful to the code that Julia actually executes.


function assembly_lines(@nospecialize(f), @nospecialize(types))
    buffer = IOBuffer()
    code_native(
        buffer, f, types;
        syntax=:intel, debuginfo=:default, binary=true, dump_module=false
    )
    return split(String(take!(buffer)), '\n'; keepempty=false)
end


###################################################### PARSING METADATA COMMENTS


using Base.Iterators: partition


const BLOCK_OPEN_REGEX = r"^; (│*)┌ @ (.*) within `(.*)`(?: @ (.*))?$"
const BLOCK_CONTINUE_REGEX = r"^; (│*) @ (.*) within `(.*)`(?: @ (.*))?$"
const BLOCK_CLOSE_REGEX = r"^; (│*)(└+)$"
const CODE_INFO_REGEX = r"^; code origin: ([0-9a-f]+), code size: ([0-9]+)$"
const HEX_INSTRUCTION_REGEX = r"^; ([0-9a-f]{4}): ([0-9a-f ]*)(?:\s*#.*)?$"


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
            line_number = parse(Int, s[colon_index+1:end]; base=10)
            return SourceLocation(s[1:colon_index-1], line_number)
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
    short_address::UInt16
    binary::Vector{UInt8}
    context::Vector{SourceContext}
end


struct AssemblyLabel
    name::SubString{String}
end


function parse_metadata(lines::Vector{SubString{String}})

    code_origin = nothing
    code_size = nothing
    last_short_address = nothing
    last_binary = nothing
    context_stack = SourceContext[]
    result = Union{AssemblyInstruction,AssemblyLabel}[]

    for line in lines

        if ((line == "\t.text") ||
            (line == "\t.section\t__TEXT,__text,regular,pure_instructions"))

            # Ignore .text section header.
            continue

        elseif startswith(line, ';')

            # The output of `code_native` contains several types of comments
            # that provide useful metadata about the generated assembly code.
            # We use regular expressions to detect and parse these comments.
            block_open_match = match(BLOCK_OPEN_REGEX, line)
            block_continue_match = match(BLOCK_CONTINUE_REGEX, line)
            block_close_match = match(BLOCK_CLOSE_REGEX, line)
            code_info_match = match(CODE_INFO_REGEX, line)
            hex_instruction_match = match(HEX_INSTRUCTION_REGEX, line)

            # Assert that at most one match occurred.
            @assert +(
                !isnothing(block_open_match),
                !isnothing(block_continue_match),
                !isnothing(block_close_match),
                !isnothing(code_info_match),
                !isnothing(hex_instruction_match),
            ) <= 1

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

            elseif !isnothing(code_info_match)

                # A code_info comment specifies the location in (virtual)
                # memory and size of the compiled machine code for a particular
                # Julia function. Exactly one code_info comment should appear
                # in the output of each call to `code_native`.
                @assert isnothing(code_origin) && isnothing(code_size)
                code_origin_str, code_size_str = code_info_match
                code_origin = parse(UInt, code_origin_str; base=16)
                code_size = parse(Int, code_size_str; base=10)

            elseif !isnothing(hex_instruction_match)

                # A hex_instruction comment specifies the binary machine code
                # representation of the following assembly instruction.
                # AssemblyView.jl makes no effort to decode this; we simply
                # store it and pass it through to user.
                short_address_str, byte_str = hex_instruction_match
                last_short_address = parse(UInt16, short_address_str; base=16)
                byte_str = replace(byte_str, isspace => "")
                @assert iseven(length(byte_str))
                last_binary = [
                    parse(UInt8, byte; base=16)
                    for byte in partition(byte_str, 2)
                ]

            else
                @warn "Ignoring assembly comment in unrecognized format: $line"
            end

        elseif startswith(line, '\t')
            @assert !isnothing(last_short_address)
            @assert !isnothing(last_binary)
            push!(result, AssemblyInstruction(
                line, last_short_address, last_binary, copy(context_stack)
            ))
        elseif endswith(line, ':')
            push!(result, AssemblyLabel(line[1:end-1]))
        else
            @warn "Ignoring assembly line in unrecognized format: $line"
        end

    end

    @assert !isnothing(code_origin)
    @assert !isnothing(code_size)
    @assert isempty(context_stack)
    return (code_origin, code_size, result)

end


########################################################### PARSING X86 ASSEMBLY


const X86_REGISTERS = Dict{String,Tuple{Symbol,Int}}([
    "al" => (:R0, 8), "ah" => (:R0, 8), "ax" => (:R0, 16), "eax" => (:R0, 32), "rax" => (:R0, 64),
    "cl" => (:R1, 8), "ch" => (:R1, 8), "cx" => (:R1, 16), "ecx" => (:R1, 32), "rcx" => (:R1, 64),
    "dl" => (:R2, 8), "dh" => (:R2, 8), "dx" => (:R2, 16), "edx" => (:R2, 32), "rdx" => (:R2, 64),
    "bl" => (:R3, 8), "bh" => (:R3, 8), "bx" => (:R3, 16), "ebx" => (:R3, 32), "rbx" => (:R3, 64),
    "spl" => (:R4, 8), "sp" => (:R4, 16), "esp" => (:R4, 32), "rsp" => (:R4, 64),
    "bpl" => (:R5, 8), "bp" => (:R5, 16), "ebp" => (:R5, 32), "rbp" => (:R5, 64),
    "sil" => (:R6, 8), "si" => (:R6, 16), "esi" => (:R6, 32), "rsi" => (:R6, 64),
    "dil" => (:R7, 8), "di" => (:R7, 16), "edi" => (:R7, 32), "rdi" => (:R7, 64),
    "r8b" => (:R8, 8), "r8w" => (:R8, 16), "r8d" => (:R8, 32), "r8" => (:R8, 64),
    "r9b" => (:R9, 8), "r9w" => (:R9, 16), "r9d" => (:R9, 32), "r9" => (:R9, 64),
    "r10b" => (:R10, 8), "r10w" => (:R10, 16), "r10d" => (:R10, 32), "r10" => (:R10, 64),
    "r11b" => (:R11, 8), "r11w" => (:R11, 16), "r11d" => (:R11, 32), "r11" => (:R11, 64),
    "r12b" => (:R12, 8), "r12w" => (:R12, 16), "r12d" => (:R12, 32), "r12" => (:R12, 64),
    "r13b" => (:R13, 8), "r13w" => (:R13, 16), "r13d" => (:R13, 32), "r13" => (:R13, 64),
    "r14b" => (:R14, 8), "r14w" => (:R14, 16), "r14d" => (:R14, 32), "r14" => (:R14, 64),
    "r15b" => (:R15, 8), "r15w" => (:R15, 16), "r15d" => (:R15, 32), "r15" => (:R15, 64),
    "xmm0" => (:SIMD0, 128), "ymm0" => (:SIMD0, 256), "zmm0" => (:SIMD0, 512),
    "xmm1" => (:SIMD1, 128), "ymm1" => (:SIMD1, 256), "zmm1" => (:SIMD1, 512),
    "xmm2" => (:SIMD2, 128), "ymm2" => (:SIMD2, 256), "zmm2" => (:SIMD2, 512),
    "xmm3" => (:SIMD3, 128), "ymm3" => (:SIMD3, 256), "zmm3" => (:SIMD3, 512),
    "xmm4" => (:SIMD4, 128), "ymm4" => (:SIMD4, 256), "zmm4" => (:SIMD4, 512),
    "xmm5" => (:SIMD5, 128), "ymm5" => (:SIMD5, 256), "zmm5" => (:SIMD5, 512),
    "xmm6" => (:SIMD6, 128), "ymm6" => (:SIMD6, 256), "zmm6" => (:SIMD6, 512),
    "xmm7" => (:SIMD7, 128), "ymm7" => (:SIMD7, 256), "zmm7" => (:SIMD7, 512),
    "xmm8" => (:SIMD8, 128), "ymm8" => (:SIMD8, 256), "zmm8" => (:SIMD8, 512),
    "xmm9" => (:SIMD9, 128), "ymm9" => (:SIMD9, 256), "zmm9" => (:SIMD9, 512),
    "xmm10" => (:SIMD10, 128), "ymm10" => (:SIMD10, 256), "zmm10" => (:SIMD10, 512),
    "xmm11" => (:SIMD11, 128), "ymm11" => (:SIMD11, 256), "zmm11" => (:SIMD11, 512),
    "xmm12" => (:SIMD12, 128), "ymm12" => (:SIMD12, 256), "zmm12" => (:SIMD12, 512),
    "xmm13" => (:SIMD13, 128), "ymm13" => (:SIMD13, 256), "zmm13" => (:SIMD13, 512),
    "xmm14" => (:SIMD14, 128), "ymm14" => (:SIMD14, 256), "zmm14" => (:SIMD14, 512),
    "xmm15" => (:SIMD15, 128), "ymm15" => (:SIMD15, 256), "zmm15" => (:SIMD15, 512),
    "xmm16" => (:SIMD16, 128), "ymm16" => (:SIMD16, 256), "zmm16" => (:SIMD16, 512),
    "xmm17" => (:SIMD17, 128), "ymm17" => (:SIMD17, 256), "zmm17" => (:SIMD17, 512),
    "xmm18" => (:SIMD18, 128), "ymm18" => (:SIMD18, 256), "zmm18" => (:SIMD18, 512),
    "xmm19" => (:SIMD19, 128), "ymm19" => (:SIMD19, 256), "zmm19" => (:SIMD19, 512),
    "xmm20" => (:SIMD20, 128), "ymm20" => (:SIMD20, 256), "zmm20" => (:SIMD20, 512),
    "xmm21" => (:SIMD21, 128), "ymm21" => (:SIMD21, 256), "zmm21" => (:SIMD21, 512),
    "xmm22" => (:SIMD22, 128), "ymm22" => (:SIMD22, 256), "zmm22" => (:SIMD22, 512),
    "xmm23" => (:SIMD23, 128), "ymm23" => (:SIMD23, 256), "zmm23" => (:SIMD23, 512),
    "xmm24" => (:SIMD24, 128), "ymm24" => (:SIMD24, 256), "zmm24" => (:SIMD24, 512),
    "xmm25" => (:SIMD25, 128), "ymm25" => (:SIMD25, 256), "zmm25" => (:SIMD25, 512),
    "xmm26" => (:SIMD26, 128), "ymm26" => (:SIMD26, 256), "zmm26" => (:SIMD26, 512),
    "xmm27" => (:SIMD27, 128), "ymm27" => (:SIMD27, 256), "zmm27" => (:SIMD27, 512),
    "xmm28" => (:SIMD28, 128), "ymm28" => (:SIMD28, 256), "zmm28" => (:SIMD28, 512),
    "xmm29" => (:SIMD29, 128), "ymm29" => (:SIMD29, 256), "zmm29" => (:SIMD29, 512),
    "xmm30" => (:SIMD30, 128), "ymm30" => (:SIMD30, 256), "zmm30" => (:SIMD30, 512),
    "xmm31" => (:SIMD31, 128), "ymm31" => (:SIMD31, 256), "zmm31" => (:SIMD31, 512),
    "k0" => (:MASK0, 64), "k1" => (:MASK1, 64), "k2" => (:MASK2, 64), "k3" => (:MASK3, 64),
    "k4" => (:MASK4, 64), "k5" => (:MASK5, 64), "k6" => (:MASK6, 64), "k7" => (:MASK7, 64),
])


const X86_POINTER_SIZES = Dict{String,Int}(
    "byte" => 8, "word" => 16, "dword" => 32, "qword" => 64,
    "tbyte" => 80, "xmmword" => 128, "ymmword" => 256, "zmmword" => 512,
)


const X86_ADDRESS_OPERAND_REGEX = r"^\[(.*)\]$"
const X86_POINTER_OPERAND_REGEX = r"^(.*) ptr \[(.*)\]$"
const X86_OFFSET_OPERAND_REGEX = r"^offset (.*)$"
const X86_LABEL_OPERAND_REGEX = r"^L[0-9]+$"
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
        # Assert that at most one match occurred.
        @assert +(
            !isnothing(address_match),
            !isnothing(pointer_match),
            !isnothing(offset_match),
            !isnothing(label_match),
            !isnothing(integer_match),
        ) <= 1
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
    short_address::UInt16
    binary::Vector{UInt8}
    context::Vector{SourceContext}
end


function X86Instruction(instruction::AssemblyInstruction)
    code = instruction.code
    @assert startswith(code, '\t')
    comment_index = findfirst('#', code)
    comment = isnothing(comment_index) ? nothing : code[comment_index:end]
    code = strip(isnothing(comment_index) ? code : code[1:comment_index-1])
    tab_index = findfirst('\t', code)
    if isnothing(tab_index)
        return X86Instruction(
            code, X86Operand[], comment,
            instruction.short_address, instruction.binary, instruction.context
        )
    else
        opcode = code[1:tab_index-1]
        operands = (opcode == "nop") ? X86Operand[] : parse_x86_operand.(
            strip.(split(code[tab_index+1:end], ','))
        )
        return X86Instruction(
            opcode, operands, comment,
            instruction.short_address, instruction.binary, instruction.context
        )
    end
end


function Base.print(io::IO, op::X86RegisterOperand)
    printstyled(io, op.id; bold=true)
    printstyled(io, '[', op.origin, ':', op.origin + op.size - 1, ']'; color=:yellow)
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


##################################################### ASSEMBLY PARSING INTERFACE


export parsed_asm


function parsed_asm(@nospecialize(f), @nospecialize(types...))
    @static if Sys.ARCH == :x86_64
        code_origin, code_size, lines = parse_metadata(assembly_lines(f, types))
        current_address = code_origin % UInt16
        current_size = 0
        result = Union{X86Instruction,AssemblyLabel}[]
        for line in lines
            if line isa AssemblyInstruction
                instruction = X86Instruction(line)
                @assert instruction.short_address == current_address
                current_address += length(instruction.binary) % UInt16
                current_size += length(instruction.binary)
                push!(result, instruction)
            elseif line isa AssemblyLabel
                push!(result, line)
            else
                @assert false
            end
        end
        @assert current_size == code_size
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
        _, _, lines = parse_metadata(assembly_lines(f, types))
        for line in lines
            if line isa AssemblyInstruction
                println(line.code)
            elseif line isa AssemblyLabel
                println(line.name, ':')
            else
                @assert false
            end
        end
    end
end


################################################################################

end # module AssemblyView
