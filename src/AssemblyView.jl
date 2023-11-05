module AssemblyView


####################################################### EXTRACTING ASSEMBLY CODE


using InteractiveUtils: code_native


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


const X86_REGISTERS = Dict{String,Tuple{Symbol,Int}}(
    "ah" => (:A, 8), "al" => (:A, 8), "ax" => (:A, 16), "eax" => (:A, 32), "rax" => (:A, 64),
    "ch" => (:C, 8), "cl" => (:C, 8), "cx" => (:C, 16), "ecx" => (:C, 32), "rcx" => (:C, 64),
    "dh" => (:D, 8), "dl" => (:D, 8), "dx" => (:D, 16), "edx" => (:D, 32), "rdx" => (:D, 64),
    "bh" => (:B, 8), "bl" => (:B, 8), "bx" => (:B, 16), "ebx" => (:B, 32), "rbx" => (:B, 64),
    "spl" => (:SP, 8), "sp" => (:SP, 16), "esp" => (:SP, 32), "rsp" => (:SP, 64),
    "bpl" => (:BP, 8), "bp" => (:BP, 16), "ebp" => (:BP, 32), "rbp" => (:BP, 64),
    "sil" => (:SI, 8), "si" => (:SI, 16), "esi" => (:SI, 32), "rsi" => (:SI, 64),
    "dil" => (:DI, 8), "di" => (:DI, 16), "edi" => (:DI, 32), "rdi" => (:DI, 64),
    "r8b" => (:R8, 8), "r8w" => (:R8, 16), "r8d" => (:R8, 32), "r8" => (:R8, 64),
    "r9b" => (:R9, 8), "r9w" => (:R9, 16), "r9d" => (:R9, 32), "r9" => (:R9, 64),
    "r10b" => (:R10, 8), "r10w" => (:R10, 16), "r10d" => (:R10, 32), "r10" => (:R10, 64),
    "r11b" => (:R11, 8), "r11w" => (:R11, 16), "r11d" => (:R11, 32), "r11" => (:R11, 64),
    "r12b" => (:R12, 8), "r12w" => (:R12, 16), "r12d" => (:R12, 32), "r12" => (:R12, 64),
    "r13b" => (:R13, 8), "r13w" => (:R13, 16), "r13d" => (:R13, 32), "r13" => (:R13, 64),
    "r14b" => (:R14, 8), "r14w" => (:R14, 16), "r14d" => (:R14, 32), "r14" => (:R14, 64),
    "r15b" => (:R15, 8), "r15w" => (:R15, 16), "r15d" => (:R15, 32), "r15" => (:R15, 64),
    "xmm0" => (:SSE0, 128), "ymm0" => (:SSE0, 256), "zmm0" => (:SSE0, 512),
    "xmm1" => (:SSE1, 128), "ymm1" => (:SSE1, 256), "zmm1" => (:SSE1, 512),
    "xmm2" => (:SSE2, 128), "ymm2" => (:SSE2, 256), "zmm2" => (:SSE2, 512),
    "xmm3" => (:SSE3, 128), "ymm3" => (:SSE3, 256), "zmm3" => (:SSE3, 512),
    "xmm4" => (:SSE4, 128), "ymm4" => (:SSE4, 256), "zmm4" => (:SSE4, 512),
    "xmm5" => (:SSE5, 128), "ymm5" => (:SSE5, 256), "zmm5" => (:SSE5, 512),
    "xmm6" => (:SSE6, 128), "ymm6" => (:SSE6, 256), "zmm6" => (:SSE6, 512),
    "xmm7" => (:SSE7, 128), "ymm7" => (:SSE7, 256), "zmm7" => (:SSE7, 512),
    "xmm8" => (:SSE8, 128), "ymm8" => (:SSE8, 256), "zmm8" => (:SSE8, 512),
    "xmm9" => (:SSE9, 128), "ymm9" => (:SSE9, 256), "zmm9" => (:SSE9, 512),
    "xmm10" => (:SSE10, 128), "ymm10" => (:SSE10, 256), "zmm10" => (:SSE10, 512),
    "xmm11" => (:SSE11, 128), "ymm11" => (:SSE11, 256), "zmm11" => (:SSE11, 512),
    "xmm12" => (:SSE12, 128), "ymm12" => (:SSE12, 256), "zmm12" => (:SSE12, 512),
    "xmm13" => (:SSE13, 128), "ymm13" => (:SSE13, 256), "zmm13" => (:SSE13, 512),
    "xmm14" => (:SSE14, 128), "ymm14" => (:SSE14, 256), "zmm14" => (:SSE14, 512),
    "xmm15" => (:SSE15, 128), "ymm15" => (:SSE15, 256), "zmm15" => (:SSE15, 512),
    "xmm16" => (:SSE16, 128), "ymm16" => (:SSE16, 256), "zmm16" => (:SSE16, 512),
    "xmm17" => (:SSE17, 128), "ymm17" => (:SSE17, 256), "zmm17" => (:SSE17, 512),
    "xmm18" => (:SSE18, 128), "ymm18" => (:SSE18, 256), "zmm18" => (:SSE18, 512),
    "xmm19" => (:SSE19, 128), "ymm19" => (:SSE19, 256), "zmm19" => (:SSE19, 512),
    "xmm20" => (:SSE20, 128), "ymm20" => (:SSE20, 256), "zmm20" => (:SSE20, 512),
    "xmm21" => (:SSE21, 128), "ymm21" => (:SSE21, 256), "zmm21" => (:SSE21, 512),
    "xmm22" => (:SSE22, 128), "ymm22" => (:SSE22, 256), "zmm22" => (:SSE22, 512),
    "xmm23" => (:SSE23, 128), "ymm23" => (:SSE23, 256), "zmm23" => (:SSE23, 512),
    "xmm24" => (:SSE24, 128), "ymm24" => (:SSE24, 256), "zmm24" => (:SSE24, 512),
    "xmm25" => (:SSE25, 128), "ymm25" => (:SSE25, 256), "zmm25" => (:SSE25, 512),
    "xmm26" => (:SSE26, 128), "ymm26" => (:SSE26, 256), "zmm26" => (:SSE26, 512),
    "xmm27" => (:SSE27, 128), "ymm27" => (:SSE27, 256), "zmm27" => (:SSE27, 512),
    "xmm28" => (:SSE28, 128), "ymm28" => (:SSE28, 256), "zmm28" => (:SSE28, 512),
    "xmm29" => (:SSE29, 128), "ymm29" => (:SSE29, 256), "zmm29" => (:SSE29, 512),
    "xmm30" => (:SSE30, 128), "ymm30" => (:SSE30, 256), "zmm30" => (:SSE30, 512),
    "xmm31" => (:SSE31, 128), "ymm31" => (:SSE31, 256), "zmm31" => (:SSE31, 512),
)


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


Base.print(io::IO, op::X86RegisterOperand) =
    print(io, op.id, '[', op.origin, ':', op.origin + op.size - 1, ']')
Base.print(io::IO, op::X86AddressOperand) = print(io, op.expr)
Base.print(io::IO, op::X86PointerOperand) =
    print(io, "*(", op.address.expr, ")[0:", op.size - 1, ']')
Base.print(io::IO, op::X86OffsetOperand) = print(io, "offset ", op.name)
Base.print(io::IO, op::X86LabelOperand) = print(io, op.name)
Base.print(io::IO, op::X86IntegerOperand) = print(io, op.value)


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


X86_PRINT_HANDLERS[("nop", DataType[])] =
    instruction::X86Instruction -> nothing


X86_PRINT_HANDLERS[("ret", DataType[])] =
    instruction::X86Instruction -> println("\treturn;")


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
                    print('\t', line.opcode, '\t', join(line.operands, ", "))
                    if !isnothing(line.comment)
                        println('\t', line.comment)
                    else
                        println()
                    end
                end
            elseif line isa AssemblyLabel
                println(line.name, ':')
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


#=

############################################################### ASSEMBLY PARSING


function asm_offsets(
        @nospecialize(func), @nospecialize(types...))::Vector{String}
    result = String[]
    stmts = parsed_asm(func, types...)
    for stmt in stmts
        if stmt isa AssemblyInstruction
            for op in stmt.operands
                if op isa AssemblyOffset
                    push!(result, op.name)
                end
            end
        end
    end
    return result
end


################################################################ PRINT UTILITIES


function assert_num_operands(instr::AssemblyInstruction, n::Int)::Nothing
    if length(instr.operands) != n
        throw(AssertionError("AssemblyView.jl INTERNAL ERROR: Encountered" *
                             " $(instr.opcode) instruction with wrong" *
                             " number of operands (expected $n; found" *
                             " $(length(instr.operands)))."))
    end
end


################################################################ IGNORED OPCODES


const X86_IGNORED_OPCODES = [
    "nop",
    "vzeroupper",
    "ud2",
]


for opcode in X86_IGNORED_OPCODES
    PRINT_HANDLERS[opcode] = (io::IO, instr::AssemblyInstruction) -> print(io)
end


################################################################ CONTROL OPCODES


const X86_CONTROL_OPCODES = [
    "push",
    "pop",
    "call",
    "ret",
    "jmp",
]


function verbatim_print_hander(io::IO, instr::AssemblyInstruction)::Nothing
    assert_num_operands(instr, 1)
    print(io, instr.opcode, ' ', instr.operands[1])
end

PRINT_HANDLERS["push"] =
PRINT_HANDLERS["pop"] =
PRINT_HANDLERS["call"] = verbatim_print_hander


PRINT_HANDLERS["ret"] = (io::IO, instr::AssemblyInstruction) -> begin
    assert_num_operands(instr, 0)
    print(io, "return")
end


PRINT_HANDLERS["jmp"] = (io::IO, instr::AssemblyInstruction) -> begin
    assert_num_operands(instr, 1)
    label = instr.operands[1]
    @assert label isa Union{AssemblyImmediate,AssemblyMemoryOperand}
    print(io, "goto $label")
end


####################################################### CONDITIONAL JUMP OPCODES


const X86_COMPARISON_OPCODES = [
    "cmp",
    "vucomisd",
]


PRINT_HANDLERS["cmp"] =
PRINT_HANDLERS["vucomisd"] = (io::IO, instr::AssemblyInstruction) -> begin
    assert_num_operands(instr, 2)
    a, b = instr.operands
    print(io, "$a <=> $b")
end


const X86_CONDITIONAL_JUMP_OPCODES = [
    "ja", "jnbe", "jae", "jnb", "jb", "jnae", "jbe", "jna",
    "jg", "jnle", "jge", "jnl", "jl", "jnge", "jle", "jng",
    "je", "jne",
]


function make_conditional_jump_handler(op::String)::Function
    return (io::IO, instr::AssemblyInstruction) -> begin
        if length(instr.operands) == 1
            label = instr.operands[1]
            print(io, "if ($op) goto $label")
        elseif length(instr.operands) == 3
            a, b, label = instr.operands
            print(io, "if ($a $op $b) goto $label")
        else
            @assert false
        end
    end
end


PRINT_HANDLERS["ja"] =
PRINT_HANDLERS["jnbe"] =
PRINT_HANDLERS["jg"] =
PRINT_HANDLERS["jnle"] = make_conditional_jump_handler(">")


PRINT_HANDLERS["jae"] =
PRINT_HANDLERS["jnb"] =
PRINT_HANDLERS["jge"] =
PRINT_HANDLERS["jnl"] = make_conditional_jump_handler(">=")


PRINT_HANDLERS["jb"] =
PRINT_HANDLERS["jnae"] =
PRINT_HANDLERS["jl"] =
PRINT_HANDLERS["jnge"] = make_conditional_jump_handler("<")


PRINT_HANDLERS["jbe"] =
PRINT_HANDLERS["jna"] =
PRINT_HANDLERS["jle"] =
PRINT_HANDLERS["jng"] = make_conditional_jump_handler("<=")


PRINT_HANDLERS["je"] = make_conditional_jump_handler("==")


PRINT_HANDLERS["jne"] = make_conditional_jump_handler("!=")


#################################################################### MOV OPCODES


const X86_MOV_OPCODES = [
    "mov",
    "movzx",
    "movabs",
    "vmovss",
    "vmovsd",
    "vmovups",
    "vmovupd",
    "vmovdqu",
    "vmovaps",
    "vmovapd",
    "vmovdqa",
    "vextractf128",
]


PRINT_HANDLERS["mov"] =
PRINT_HANDLERS["movzx"] =
PRINT_HANDLERS["movabs"] =
PRINT_HANDLERS["vmovss"] =
PRINT_HANDLERS["vmovsd"] = (io::IO, instr::AssemblyInstruction) -> begin
    assert_num_operands(instr, 2)
    dst, src = instr.operands
    print(io, "$dst = $src")
end


PRINT_HANDLERS["vmovups"] =
PRINT_HANDLERS["vmovupd"] =
PRINT_HANDLERS["vmovdqu"] = (io::IO, instr::AssemblyInstruction) -> begin
    assert_num_operands(instr, 2)
    dst, src = instr.operands
    print(io, "$dst .= $src")
end


PRINT_HANDLERS["vmovaps"] =
PRINT_HANDLERS["vmovapd"] =
PRINT_HANDLERS["vmovdqa"] = (io::IO, instr::AssemblyInstruction) -> begin
    assert_num_operands(instr, 2)
    dst, src = instr.operands
    print(io, "$dst .= $src [aligned]")
end


PRINT_HANDLERS["vextractf128"] = (io::IO, instr::AssemblyInstruction) -> begin
    assert_num_operands(instr, 3)
    dst, src, imm = instr.operands
    @assert imm isa AssemblyImmediate
    if imm.value == "0"
        print(io, "$dst = $src[0:127]")
    elseif imm.value == "1"
        print(io, "$dst = $src[128:255]")
    else
        @assert false
    end
end


############################################################# ARITHMETIC OPCODES


const X86_ARITHMETIC_OPCODES = [
    "inc",
    "dec",
    "add",
    "sub",
    "and",
    "andn",
    "xor",
    "vxorps",
    "vxorpd",
    "shl",
    "sar",
    "lea",
    "vcvtsi2sd",
    "vaddss",
    "vaddsd",
    "vaddps",
    "vaddpd",
    "vsubss",
    "vsubsd",
    "vsubps",
    "vsubpd",
    "vmulss",
    "vmulsd",
    "vmulps",
    "vmulpd",
    "vdivss",
    "vdivsd",
    "vdivps",
    "vdivpd",
    "vpermilpd",
    "vfmadd213sd",
    "vfmadd213pd",
    "vfmadd231sd",
    "vfmadd231pd",
    "vfmsub213sd",
    "vfmsub213pd",
    "vfmsub132sd",
    "vfmsub132pd",
    "vfnmsub213sd",
    "vfnmsub213pd",
]


PRINT_HANDLERS["inc"] = (io::IO, instr::AssemblyInstruction) -> begin
    assert_num_operands(instr, 1)
    dst = instr.operands[1]
    print(io, "$dst++")
end

PRINT_HANDLERS["dec"] = (io::IO, instr::AssemblyInstruction) -> begin
    assert_num_operands(instr, 1)
    dst = instr.operands[1]
    print(io, "$dst--")
end


PRINT_HANDLERS["add"] = (io::IO, instr::AssemblyInstruction) -> begin
    assert_num_operands(instr, 2)
    dst, src = instr.operands
    print(io, "$dst += $src")
end

PRINT_HANDLERS["sub"] = (io::IO, instr::AssemblyInstruction) -> begin
    assert_num_operands(instr, 2)
    dst, src = instr.operands
    print(io, "$dst -= $src")
end


PRINT_HANDLERS["and"] = (io::IO, instr::AssemblyInstruction) -> begin
    assert_num_operands(instr, 2)
    dst, src = instr.operands
    print(io, "$dst &= $src")
end

PRINT_HANDLERS["andn"] = (io::IO, instr::AssemblyInstruction) -> begin
    assert_num_operands(instr, 3)
    dst, a, b = instr.operands
    print(io, "$dst = ~$a & $b")
end


PRINT_HANDLERS["xor"] = (io::IO, instr::AssemblyInstruction) -> begin
    assert_num_operands(instr, 2)
    dst, src = instr.operands
    if dst == src
        print(io, "$dst = 0")
    else
        print(io, "$dst ^= src")
    end
end


PRINT_HANDLERS["vxorps"] =
PRINT_HANDLERS["vxorpd"] = (io::IO, instr::AssemblyInstruction) -> begin
    assert_num_operands(instr, 3)
    dst, a, b = instr.operands
    if a == b
        print(io, "$dst .= 0")
    else
        print(io, "$dst .= $a .^ $b")
    end
end


PRINT_HANDLERS["shl"] = (io::IO, instr::AssemblyInstruction) -> begin
    assert_num_operands(instr, 2)
    dst, src = instr.operands
    print(io, "$dst <<= $src")
end


PRINT_HANDLERS["sar"] = (io::IO, instr::AssemblyInstruction) -> begin
    if length(instr.operands) == 1
        dst = instr.operands[1]
        print(io, "$dst >>= 1")
    elseif length(instr.operands) == 2
        dst, src = instr.operands
        print(io, "$dst >>= $src")
    else
        @assert false
    end
end


PRINT_HANDLERS["lea"] = (io::IO, instr::AssemblyInstruction) -> begin
    assert_num_operands(instr, 2)
    dst, src = instr.operands
    @assert dst isa AssemblyRegister
    @assert src isa AssemblyImmediate
    @assert startswith(src.value, '[')
    @assert endswith(src.value, ']')
    print(io, "$dst = $(src.value[2:end-1])")
end


PRINT_HANDLERS["vcvtsi2sd"] = (io::IO, instr::AssemblyInstruction) -> begin
    assert_num_operands(instr, 3)
    dst, a, b = instr.operands
    print(io, "$dst = (double) $b[0], $a[1]")
end


PRINT_HANDLERS["vaddss"] =
PRINT_HANDLERS["vaddsd"] = (io::IO, instr::AssemblyInstruction) -> begin
    assert_num_operands(instr, 3)
    dst, a, b = instr.operands
    print(io, "$dst = $a + $b")
end

PRINT_HANDLERS["vaddps"] =
PRINT_HANDLERS["vaddpd"] = (io::IO, instr::AssemblyInstruction) -> begin
    assert_num_operands(instr, 3)
    dst, a, b = instr.operands
    print(io, "$dst .= $a .+ $b")
end

PRINT_HANDLERS["vsubss"] =
PRINT_HANDLERS["vsubsd"] = (io::IO, instr::AssemblyInstruction) -> begin
    assert_num_operands(instr, 3)
    dst, a, b = instr.operands
    print(io, "$dst = $a - $b")
end

PRINT_HANDLERS["vsubps"] =
PRINT_HANDLERS["vsubpd"] = (io::IO, instr::AssemblyInstruction) -> begin
    assert_num_operands(instr, 3)
    dst, a, b = instr.operands
    print(io, "$dst .= $a .- $b")
end

PRINT_HANDLERS["vmulss"] =
PRINT_HANDLERS["vmulsd"] = (io::IO, instr::AssemblyInstruction) -> begin
    assert_num_operands(instr, 3)
    dst, a, b = instr.operands
    print(io, "$dst = $a * $b")
end

PRINT_HANDLERS["vmulps"] =
PRINT_HANDLERS["vmulpd"] = (io::IO, instr::AssemblyInstruction) -> begin
    assert_num_operands(instr, 3)
    dst, a, b = instr.operands
    print(io, "$dst .= $a .* $b")
end

PRINT_HANDLERS["vdivss"] =
PRINT_HANDLERS["vdivsd"] = (io::IO, instr::AssemblyInstruction) -> begin
    assert_num_operands(instr, 3)
    dst, a, b = instr.operands
    print(io, "$dst = $a / $b")
end

PRINT_HANDLERS["vdivps"] =
PRINT_HANDLERS["vdivpd"] = (io::IO, instr::AssemblyInstruction) -> begin
    assert_num_operands(instr, 3)
    dst, a, b = instr.operands
    print(io, "$dst .= $a ./ $b")
end


function comment_print_handler(io::IO, instr::AssemblyInstruction)
    @assert !isempty(instr.comment)
    parts = split(instr.comment, " = ")
    @assert length(parts) == 2
    lhs, rhs = parts
    print(io, "$lhs = $rhs")
end

PRINT_HANDLERS["vpermilpd"] =
PRINT_HANDLERS["vfmadd213sd"] =
PRINT_HANDLERS["vfmadd213pd"] =
PRINT_HANDLERS["vfmadd231sd"] =
PRINT_HANDLERS["vfmadd231pd"] =
PRINT_HANDLERS["vfmsub213sd"] =
PRINT_HANDLERS["vfmsub213pd"] =
PRINT_HANDLERS["vfmsub132sd"] =
PRINT_HANDLERS["vfmsub132pd"] =
PRINT_HANDLERS["vfnmsub213sd"] =
PRINT_HANDLERS["vfnmsub213pd"] = comment_print_handler


############################################################### COVERAGE TESTING


@assert isempty(symdiff(
    Set{String}(keys(PRINT_HANDLERS)),
    union(
        Set{String}(X86_IGNORED_OPCODES),
        Set{String}(X86_CONTROL_OPCODES),
        Set{String}(X86_COMPARISON_OPCODES),
        Set{String}(X86_CONDITIONAL_JUMP_OPCODES),
        Set{String}(X86_MOV_OPCODES),
        Set{String}(X86_ARITHMETIC_OPCODES),
    )
))


######################################################### ASSEMBLY PREPROCESSING


is_opcode(stmt::AssemblyStatement, instr::AbstractString)::Bool =
    (stmt isa AssemblyInstruction) && (stmt.opcode == instr)


function remove_nops(
        stmts::Vector{AssemblyStatement})::Vector{AssemblyStatement}
    result = AssemblyStatement[]
    for stmt in stmts
        if all(!is_opcode(stmt, nop) for nop in X86_IGNORED_OPCODES)
            push!(result, stmt)
        end
    end
    return result
end


function remove_prologue_epilogue(
        stmts::Vector{AssemblyStatement})::Vector{AssemblyStatement}

    # Check whether the first two instructions are:
    #     push rbp
    #     mov rbp, rsp
    # Exit early if this is not the case.
    if ((length(stmts) < 2)
        || !is_opcode(stmts[1], "push")
        || (length(stmts[1].operands) != 1)
        || (stmts[1].operands[1] != AssemblyRegister("rbp"))
        || !is_opcode(stmts[2], "mov")
        || (stmts[2].operands != [AssemblyRegister("rbp"),
                                  AssemblyRegister("rsp")]))
        return copy(stmts)
    end

    # At this point, we know we have at least two prologue instructions.
    # The rest of the prologue consists of "push register" instructions...
    prologue_len = 3
    saved_regs = [AssemblyRegister("rbp")]
    while ((prologue_len <= length(stmts))
           && is_opcode(stmts[prologue_len], "push")
           && (length(stmts[prologue_len].operands) == 1)
           && (stmts[prologue_len].operands[1] isa AssemblyRegister))
        push!(saved_regs, stmts[prologue_len].operands[1])
        prologue_len += 1
    end
    reverse!(saved_regs)

    # ...possibly followed by "add/sub rsp, immediate".
    if ((prologue_len <= length(stmts))
        && (is_opcode(stmts[prologue_len], "add")
            || is_opcode(stmts[prologue_len], "sub"))
        && (length(stmts[prologue_len].operands) == 2)
        && (stmts[prologue_len].operands[1] isa AssemblyRegister)
        && (stmts[prologue_len].operands[1].name == "rsp")
        && (stmts[prologue_len].operands[2] isa AssemblyImmediate))
    else
        prologue_len -= 1
    end

    # Delete all prologue instructions.
    deletion_indices = collect(1 : prologue_len)

    # To identify the epilogue, we search for all occurences of "ret".
    ret_indices = [i for i = 1 : length(stmts) if is_opcode(stmts[i], "ret")]

    # We scan backwards from each occurrence, looking for a sequence of "pop"
    # instructions that matches the "push" sequence from the prologue.
    for i in ret_indices
        epilogue = stmts[i-length(saved_regs) : i-1]
        for (stmt, reg) in zip(epilogue, saved_regs)
            if (!is_opcode(stmt, "pop")
                || (length(stmt.operands) != 1)
                || (stmt.operands[1] != reg))
                return copy(stmts)
            end
        end

        # As before, the epilogue might optionally be prefixed with an
        # "add/sub rsp, immediate" instruction. Delete this if it exists.
        j = i - length(saved_regs) - 1
        if ((j > 0)
            && (is_opcode(stmts[j], "add")
                || is_opcode(stmts[j], "sub"))
            && (length(stmts[j].operands) == 2)
            && (stmts[j].operands[1] isa AssemblyRegister)
            && (stmts[j].operands[1].name == "rsp")
            && (stmts[j].operands[2] isa AssemblyImmediate))
            append!(deletion_indices, j : i-1)
        elseif ((j > 0)
                && (is_opcode(stmts[j], "lea"))
                && (length(stmts[j].operands) == 2)
                && (stmts[j].operands[1] isa AssemblyRegister)
                && (stmts[j].operands[1].name == "rsp")
                && (stmts[j].operands[2] isa AssemblyImmediate)
                && startswith(stmts[j].operands[2].value, "[rbp"))
            append!(deletion_indices, j : i-1)
        else
            append!(deletion_indices, j+1 : i-1)
        end
    end

    return deleteat!(copy(stmts), deletion_indices)
end


function fuse_conditional_jumps(
        stmts::Vector{AssemblyStatement})::Vector{AssemblyStatement}
    stmts = copy(stmts)
    for i = 1 : length(stmts)
        if (stmts[i] isa AssemblyInstruction
            && stmts[i].opcode in X86_CONDITIONAL_JUMP_OPCODES
            && stmts[i-1] isa AssemblyInstruction
            && stmts[i-1].opcode in X86_COMPARISON_OPCODES)
            prepend!(stmts[i].operands, stmts[i-1].operands)
            stmts[i-1] = AssemblyInstruction("nop")
        end
    end
    return remove_nops(stmts)
end


is_register_arithmetic_instruction(stmt::AssemblyStatement)::Bool = (
    stmt isa AssemblyInstruction
    && (stmt.opcode in AssemblyView.X86_ARITHMETIC_OPCODES
        || stmt.opcode in AssemblyView.X86_MOV_OPCODES)
    && all(((op isa AssemblyRegister)
            || (op isa AssemblyImmediate))
           for op in stmt.operands))


is_memory_arithmetic_instruction(stmt::AssemblyStatement)::Bool = (
    stmt isa AssemblyInstruction
    && (stmt.opcode in AssemblyView.X86_ARITHMETIC_OPCODES
        || stmt.opcode in AssemblyView.X86_MOV_OPCODES)
    && all(((op isa AssemblyRegister)
            || (op isa AssemblyImmediate)
            || (op isa AssemblyMemoryOperand))
           for op in stmt.operands))


function fuse_instructions(predicate, pseudo_opcode::String, min_length::Int,
        stmts::Vector{AssemblyStatement})::Vector{AssemblyStatement}
    stmts = copy(stmts)
    i = 1
    while i <= length(stmts)
        if predicate(stmts[i])
            j = i
            affected_operands = Set{AssemblyOperand}()
            while predicate(stmts[j])
                for op in stmts[j].operands
                    if !(op isa AssemblyImmediate)
                        push!(affected_operands, op)
                    end
                end
                j += 1
            end
            j -= 1
            if (j - i + 1) >= min_length
                stmts[i] = AssemblyInstruction(pseudo_opcode,
                    push!(collect(affected_operands),
                        AssemblyImmediate(string(length(affected_operands)))))
                for k = i+1 : j
                    stmts[k] = AssemblyInstruction("nop")
                end
            end
            i = j + 1
        else
            i += 1
        end
    end
    return remove_nops(stmts)
end


################################################################ ASSEMBLY OUTPUT


const INSTRUCTION_PREFIX = "\t"
const INSTRUCTION_SUFFIX = ";"


function view_asm(io::IO, @nospecialize(func), @nospecialize(types...))::Nothing

    parsed_stmts = parsed_asm(func, types...)
    parsed_stmts = remove_nops(parsed_stmts)
    parsed_stmts = remove_prologue_epilogue(parsed_stmts)
    parsed_stmts = fuse_conditional_jumps(parsed_stmts)

    unknown_opcodes = Dict{String,Int}()
    for stmt in parsed_stmts
        if stmt isa AssemblyInstruction
            println(io, INSTRUCTION_PREFIX, stmt, INSTRUCTION_SUFFIX)
            if !haskey(PRINT_HANDLERS, stmt.opcode)
                if haskey(unknown_opcodes, stmt.opcode)
                    unknown_opcodes[stmt.opcode] += 1
                else
                    unknown_opcodes[stmt.opcode] = 1
                end
            end
        else
            println(io, stmt)
        end
    end

    frequency_table = [(count, opcode) for (opcode, count) in unknown_opcodes]
    sort!(frequency_table, rev=true)

    if !isempty(frequency_table)
        println(io)
        println(io, "Unknown opcodes:")
        for (count, opcode) in frequency_table
            println(io, INSTRUCTION_PREFIX, "$opcode ($count occurrences)")
        end
    end

    println(io)
    return nothing
end


view_asm(@nospecialize(func), @nospecialize(types...))::Nothing =
    view_asm(stdout, func, types...)

=#

end # module AssemblyView
