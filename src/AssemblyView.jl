module AssemblyView

export AssemblyOperand, AssemblyRegister, AssemblyMemoryOperand,
    AssemblyImmediate, AssemblyOffset, AssemblyStatement, AssemblyComment,
    AssemblyLabel, AssemblyInstruction,
    parsed_asm, asm_offsets, view_asm

################################################################################

using InteractiveUtils: _dump_function

################################################################################

const X86_REGISTER_NAMES = [
    "rax", "rcx", "rdx", "rbx", "rsp", "rbp", "rsi", "rdi",
    "eax", "ecx", "edx", "ebx", "esp", "ebp", "esi", "edi",
     "ax",  "cx",  "dx",  "bx",  "sp",  "bp",  "si",  "di",
     "ah",  "ch",  "dh",  "bh",
     "al",  "cl",  "dl",  "bl", "spl", "bpl", "sil", "dil",
    "r8",  "r9",  "r10",  "r11",  "r12",  "r13",  "r14",  "r15",
    "r8d", "r9d", "r10d", "r11d", "r12d", "r13d", "r14d", "r15d",
    "r8w", "r9w", "r10w", "r11w", "r12w", "r13w", "r14w", "r15w",
    "r8b", "r9b", "r10b", "r11b", "r12b", "r13b", "r14b", "r15b",
    "zmm0", "ymm0", "xmm0", "zmm1", "ymm1", "xmm1",
    "zmm2", "ymm2", "xmm2", "zmm3", "ymm3", "xmm3",
    "zmm4", "ymm4", "xmm4", "zmm5", "ymm5", "xmm5",
    "zmm6", "ymm6", "xmm6", "zmm7", "ymm7", "xmm7",
    "zmm8", "ymm8", "xmm8", "zmm9", "ymm9", "xmm9",
    "zmm10", "ymm10", "xmm10", "zmm11", "ymm11", "xmm11",
    "zmm12", "ymm12", "xmm12", "zmm13", "ymm13", "xmm13",
    "zmm14", "ymm14", "xmm14", "zmm15", "ymm15", "xmm15",
    "zmm16", "ymm16", "xmm16", "zmm17", "ymm17", "xmm17",
    "zmm18", "ymm18", "xmm18", "zmm19", "ymm19", "xmm19",
    "zmm20", "ymm20", "xmm20", "zmm21", "ymm21", "xmm21",
    "zmm22", "ymm22", "xmm22", "zmm23", "ymm23", "xmm23",
    "zmm24", "ymm24", "xmm24", "zmm25", "ymm25", "xmm25",
    "zmm26", "ymm26", "xmm26", "zmm27", "ymm27", "xmm27",
    "zmm28", "ymm28", "xmm28", "zmm29", "ymm29", "xmm29",
    "zmm30", "ymm30", "xmm30", "zmm31", "ymm31", "xmm31",
]

################################################################################

abstract type AssemblyOperand end

struct AssemblyRegister <: AssemblyOperand
    name::String
end

struct AssemblyMemoryOperand <: AssemblyOperand
    type::String
    address::String
end

struct AssemblyImmediate <: AssemblyOperand
    value::String
end

struct AssemblyOffset <: AssemblyOperand
    name::String
end

################################################################################

abstract type AssemblyStatement end

struct AssemblyComment <: AssemblyStatement
    contents::String
end

struct AssemblyLabel <: AssemblyStatement
    name::String
end

struct AssemblyInstruction <: AssemblyStatement
    instruction::String
    operands::Vector{AssemblyOperand}
    comment::String
end

AssemblyInstruction(instruction) =
    AssemblyInstruction(instruction, AssemblyOperand[], "")
AssemblyInstruction(instruction, operands) =
    AssemblyInstruction(instruction, operands, "")

################################################################################

function parse_assembly_operand(op::AbstractString)
    op = strip(op)
    if any(op == reg_name for reg_name in X86_REGISTER_NAMES)
        return AssemblyRegister(op)
    end
    mem_op_match = match(r"([a-z]+) ptr \[(.*)\]", op)
    if !isnothing(mem_op_match)
        return AssemblyMemoryOperand(mem_op_match[1], mem_op_match[2])
    end
    offset_match = match(r"offset (.*)", op)
    if !isnothing(offset_match)
        return AssemblyOffset(offset_match[1])
    end
    # TODO: Figure out how to parse immediates
    return AssemblyImmediate(op)
end

function parse_assembly_statement(stmt::AbstractString)
    # return early for comments and empty lines
    if startswith(strip(stmt), ';')
        return AssemblyComment(stmt)
    end
    if (stmt == "\t.text") || isempty(strip(stmt))
        return nothing
    end
    # tabs separate major parts of an instruction
    tokens = split(stmt, '\t')
    # if the line does not begin with a tab, it is a label
    if length(tokens) == 1
        @assert endswith(tokens[1], ':')
        label_name = tokens[1][1:end-1]
        @assert !isempty(label_name)
        return AssemblyLabel(label_name)
    # if the line does begin with a tab, it is an instruction
    else
        @assert length(tokens) > 1
        @assert isempty(tokens[1])
        # if there is no second tab, then the instruction takes no operands
        if length(tokens) == 2
            return AssemblyInstruction(tokens[2])
        # if there is a second tab, then the instruction operands follow it
        else
            @assert length(tokens) == 3
            # some instructions are output with a #-delimited comment
            arg_tokens = split(tokens[3], " # ")
            if length(arg_tokens) == 1
                return AssemblyInstruction(tokens[2],
                    parse_assembly_operand.(split(arg_tokens[1], ',')))
            elseif length(arg_tokens) == 2
                return AssemblyInstruction(tokens[2],
                    parse_assembly_operand.(split(arg_tokens[1], ',')),
                    strip(arg_tokens[2]))
            end
        end
    end
    error("cannot parse assembly statement: ", stmt)
end

################################################################################

function parsed_asm(@nospecialize(func), @nospecialize(types...);
        keep_comments::Bool=false)::Vector{AssemblyStatement}
    if (length(types) == 1) && !(types[1] isa Type)
        types = types[1]
    end
    stmts = split(_dump_function(func, types,
        true,   # Generate native code (as opposed to LLVM IR).
        false,  # Don't generate wrapper code.
        true,   # (strip_ir_metadata) Ignored when dumping native code.
        true,   # (dump_module) Ignored when dumping native code.
        :intel, # I prefer Intel assembly syntax.
        true,   # (optimize) Ignored when dumping native code.
        :source # TODO: What does debuginfo=:source mean?
    ), '\n')
    parsed_stmts = AssemblyStatement[]
    for stmt in stmts
        parsed_stmt = parse_assembly_statement(stmt)
        if !isnothing(parsed_stmt) && (keep_comments ||
                                       !(parsed_stmt isa AssemblyComment))
            push!(parsed_stmts, parsed_stmt)
        end
    end
    return parsed_stmts
end

################################################################################

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

################################################################################

print_asm(io::IO, reg::AssemblyRegister) = print(io, reg.name)
print_asm(io::IO, memop::AssemblyMemoryOperand) =
    print(io, "*(($(memop.type) *) ($(memop.address)))")
print_asm(io::IO, immed::AssemblyImmediate) = print(io, immed.value)
print_asm(io::IO, offset::AssemblyOffset) = print(io, "offset ", offset.name)

################################################################################

const PRINT_HANDLERS = Dict{String,Function}()

function assert_num_operands(instr::AssemblyInstruction, n::Int)
    if length(instr.operands) != n
        throw(AssertionError("wrong number of operands"))
    end
end

function mov_print_handler(io::IO, instr::AssemblyInstruction)
    assert_num_operands(instr, 2)
    dst, src = instr.operands
    print(io, '\t')
    if dst isa AssemblyRegister
        print(io, dst.name)
    elseif dst isa AssemblyMemoryOperand
        print(io, "*(", dst.address, ")")
    else
        @assert false "mov destination must be register or memory"
    end
    print(io, " = ")
    if src isa AssemblyMemoryOperand
        print(io, "*(", src.address, ")")
    else
        print_asm(io, src)
    end
    println(io, ';')
end

PRINT_HANDLERS["mov"    ] = mov_print_handler
PRINT_HANDLERS["movabs" ] = mov_print_handler
PRINT_HANDLERS["vmovss" ] = mov_print_handler
PRINT_HANDLERS["vmovsd" ] = mov_print_handler
PRINT_HANDLERS["vmovaps"] = mov_print_handler
PRINT_HANDLERS["vmovups"] = mov_print_handler
PRINT_HANDLERS["vmovapd"] = mov_print_handler
PRINT_HANDLERS["vmovupd"] = mov_print_handler

function lea_print_handler(io::IO, instr::AssemblyInstruction)
    assert_num_operands(instr, 2)
    dst, src = instr.operands
    @assert dst isa AssemblyRegister
    @assert src isa AssemblyImmediate
    @assert startswith(src.value, '[')
    @assert endswith(src.value, ']')
    println(io, '\t', dst.name, " = ", src.value[2:end-1], ';')
end

PRINT_HANDLERS["lea"] = lea_print_handler

function jmp_print_handler(io::IO, instr::AssemblyInstruction)
    assert_num_operands(instr, 1)
    dst = instr.operands[1]
    @assert dst isa AssemblyImmediate
    println(io, "\tgoto ", dst.value, ';')
end

PRINT_HANDLERS["jmp"] = jmp_print_handler

function ret_print_handler(io::IO, instr::AssemblyInstruction)
    assert_num_operands(instr, 0)
    println(io, "\treturn;")
end

PRINT_HANDLERS["ret"] = ret_print_handler

function xor_print_handler(io::IO, instr::AssemblyInstruction)
    if length(instr.operands) == 2
        dst, src = instr.operands
        if dst == src
            print(io, '\t')
            print_asm(io, dst)
            println(io, " = 0;")
        else
            print(io, '\t')
            print_asm(io, dst)
            print(io, " ^= ")
            print_asm(io, src)
            println(io, ";")
        end
    elseif length(instr.operands) == 3
        dst, src1, src2 = instr.operands
        if (dst == src1) && (src1 == src2)
            print(io, '\t')
            print_asm(io, dst)
            println(io, " = 0;")
        else
            @assert false
        end
    else
        @assert false
    end
end

PRINT_HANDLERS["xor"   ] = xor_print_handler
PRINT_HANDLERS["vxorss"] = xor_print_handler
PRINT_HANDLERS["vxorsd"] = xor_print_handler
PRINT_HANDLERS["vxorps"] = xor_print_handler
PRINT_HANDLERS["vxorpd"] = xor_print_handler

function add_print_handler(io::IO, instr::AssemblyInstruction)
    if length(instr.operands) == 2
        dst, src = instr.operands
        print(io, '\t')
        print_asm(io, dst)
        print(io, " += ")
        print_asm(io, src)
        println(io, ";")
    elseif length(instr.operands) == 3
        dst, src1, src2 = instr.operands
        if dst == src1
            print(io, '\t')
            print_asm(io, dst)
            print(io, " += ")
            print_asm(io, src2)
            println(io, ";")
        elseif dst == src2
            print(io, '\t')
            print_asm(io, dst)
            print(io, " += ")
            print_asm(io, src1)
            println(io, ";")
        else
            print(io, '\t')
            print_asm(io, dst)
            print(io, " = ")
            print_asm(io, src1)
            print(io, " + ")
            print_asm(io, src2)
            println(io, ";")
        end
    else
        @assert false
    end
end

PRINT_HANDLERS["add"   ] = add_print_handler
PRINT_HANDLERS["vaddss"] = add_print_handler
PRINT_HANDLERS["vaddsd"] = add_print_handler
PRINT_HANDLERS["vaddps"] = add_print_handler
PRINT_HANDLERS["vaddpd"] = add_print_handler

function unknown_print_handler(io::IO, instr::AssemblyInstruction)
    print(io, '\t', rpad('{' * instr.instruction * '}', 16))
    for (i, op) in enumerate(instr.operands)
        if i > 1
            print(io, ", ")
        end
        print_asm(io, op)
    end
    if !isempty(instr.comment)
        print(io, " // ", instr.comment)
    end
    println(io)
end

################################################################################

function println_asm(io::IO, comment::AssemblyComment)::Bool
    println(io, comment.contents)
    return true
end

function println_asm(io::IO, label::AssemblyLabel)::Bool
    println(io, label.name, ':')
    return true
end

function println_asm(io::IO, instr::AssemblyInstruction)::Bool
    instr_lower = lowercase(instr.instruction)
    if haskey(PRINT_HANDLERS, instr_lower)
        try
            PRINT_HANDLERS[instr_lower](io, instr)
            return true
        catch e
            if e isa AssertionError
                unknown_print_handler(io, instr)
                return false
            else
                rethrow(e)
            end
        end
    else
        unknown_print_handler(io, instr)
        return false
    end
end

################################################################################

is_instr(stmt::AssemblyStatement, instr::AbstractString)::Bool =
    (stmt isa AssemblyInstruction) && (stmt.instruction == instr)

function remove_nops(
        stmts::Vector{AssemblyStatement})::Vector{AssemblyStatement}
    result = AssemblyStatement[]
    for stmt in stmts
        if all(!is_instr(stmt, nop) for nop in ["nop", "vzeroupper", "ud2"])
            push!(result, stmt)
        end
    end
    return result
end

function remove_prologue_epilogue(
        stmts::Vector{AssemblyStatement})::Vector{AssemblyStatement}
    if (length(stmts) == 0) || !is_instr(stmts[1], "push")
        return copy(stmts)
    end
    @assert length(stmts[1].operands) == 1
    @assert stmts[1].operands[1] == AssemblyRegister("rbp")
    @assert is_instr(stmts[2], "mov")
    @assert stmts[2].operands == [AssemblyRegister("rbp"),
                                  AssemblyRegister("rsp")]
    prologue_len = 3
    while is_instr(stmts[prologue_len], "push")
        prologue_len += 1
    end
    prologue_len -= 1
    saved_regs = [AssemblyRegister("rbp")]
    for j = 3 : prologue_len
        @assert length(stmts[j].operands) == 1
        @assert stmts[j].operands[1] isa AssemblyRegister
        push!(saved_regs, stmts[j].operands[1])
    end
    reverse!(saved_regs)
    ret_indices = [i for i = 1 : length(stmts) if is_instr(stmts[i], "ret")]
    for i in ret_indices
        epilogue = stmts[i-length(saved_regs) : i-1]
        for (stmt, reg) in zip(epilogue, saved_regs)
            @assert is_instr(stmt, "pop")
            @assert length(stmt.operands) == 1
            @assert stmt.operands[1] == reg
        end
    end
    return deleteat!(copy(stmts), vcat(1:prologue_len,
        [i-length(saved_regs) : i-1 for i in ret_indices]...))
end

function view_asm(io::IO, @nospecialize(func), @nospecialize(types...))::Nothing
    parsed_stmts = parsed_asm(func, types...)
    parsed_stmts = remove_nops(parsed_stmts)
    try
        parsed_stmts = remove_prologue_epilogue(parsed_stmts)
    catch e
        if !(e isa AssertionError)
            rethrow(e)
        end
        @warn "Failed to remove prologue and epilogue from function $func"
    end
    for stmt in parsed_stmts
        println_asm(io, stmt)
    end
end

view_asm(@nospecialize(func), @nospecialize(types...))::Nothing =
    view_asm(stdout, func, types...)

end # module AssemblyView
