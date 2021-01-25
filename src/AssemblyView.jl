module AssemblyView

export AssemblyOperand, AssemblyRegister, AssemblyMemoryOperand,
    AssemblyImmediate, AssemblyOffset, AssemblyStatement, AssemblyComment,
    AssemblyLabel, AssemblyInstruction,
    parsed_asm, asm_offsets, view_asm

using InteractiveUtils: _dump_function


################################################################# REGISTER NAMES


const X86_REGISTER_NAMES = Dict{String,Symbol}(
    "ah" => :A, "al" => :A, "ax" => :A, "eax" => :A, "rax" => :A,
    "ch" => :C, "cl" => :C, "cx" => :C, "ecx" => :C, "rcx" => :C,
    "dh" => :D, "dl" => :D, "dx" => :D, "edx" => :D, "rdx" => :D,
    "bh" => :B, "bl" => :B, "bx" => :B, "ebx" => :B, "rbx" => :B,
    "spl" => :SP, "sp" => :SP, "esp" => :SP, "rsp" => :SP,
    "bpl" => :BP, "bp" => :BP, "ebp" => :BP, "rbp" => :BP,
    "sil" => :SI, "si" => :SI, "esi" => :SI, "rsi" => :SI,
    "dil" => :DI, "di" => :DI, "edi" => :DI, "rdi" => :DI,
    "r8b" => :R8, "r8w" => :R8, "r8d" => :R8, "r8" => :R8,
    "r9b" => :R9, "r9w" => :R9, "r9d" => :R9, "r9" => :R9,
    "r10b" => :R10, "r10w" => :R10, "r10d" => :R10, "r10" => :R10,
    "r11b" => :R11, "r11w" => :R11, "r11d" => :R11, "r11" => :R11,
    "r12b" => :R12, "r12w" => :R12, "r12d" => :R12, "r12" => :R12,
    "r13b" => :R13, "r13w" => :R13, "r13d" => :R13, "r13" => :R13,
    "r14b" => :R14, "r14w" => :R14, "r14d" => :R14, "r14" => :R14,
    "r15b" => :R15, "r15w" => :R15, "r15d" => :R15, "r15" => :R15,
    "xmm0" => :SSE0, "ymm0" => :SSE0, "zmm0" => :SSE0,
    "xmm1" => :SSE1, "ymm1" => :SSE1, "zmm1" => :SSE1,
    "xmm2" => :SSE2, "ymm2" => :SSE2, "zmm2" => :SSE2,
    "xmm3" => :SSE3, "ymm3" => :SSE3, "zmm3" => :SSE3,
    "xmm4" => :SSE4, "ymm4" => :SSE4, "zmm4" => :SSE4,
    "xmm5" => :SSE5, "ymm5" => :SSE5, "zmm5" => :SSE5,
    "xmm6" => :SSE6, "ymm6" => :SSE6, "zmm6" => :SSE6,
    "xmm7" => :SSE7, "ymm7" => :SSE7, "zmm7" => :SSE7,
    "xmm8" => :SSE8, "ymm8" => :SSE8, "zmm8" => :SSE8,
    "xmm9" => :SSE9, "ymm9" => :SSE9, "zmm9" => :SSE9,
    "xmm10" => :SSE10, "ymm10" => :SSE10, "zmm10" => :SSE10,
    "xmm11" => :SSE11, "ymm11" => :SSE11, "zmm11" => :SSE11,
    "xmm12" => :SSE12, "ymm12" => :SSE12, "zmm12" => :SSE12,
    "xmm13" => :SSE13, "ymm13" => :SSE13, "zmm13" => :SSE13,
    "xmm14" => :SSE14, "ymm14" => :SSE14, "zmm14" => :SSE14,
    "xmm15" => :SSE15, "ymm15" => :SSE15, "zmm15" => :SSE15,
    "xmm16" => :SSE16, "ymm16" => :SSE16, "zmm16" => :SSE16,
    "xmm17" => :SSE17, "ymm17" => :SSE17, "zmm17" => :SSE17,
    "xmm18" => :SSE18, "ymm18" => :SSE18, "zmm18" => :SSE18,
    "xmm19" => :SSE19, "ymm19" => :SSE19, "zmm19" => :SSE19,
    "xmm20" => :SSE20, "ymm20" => :SSE20, "zmm20" => :SSE20,
    "xmm21" => :SSE21, "ymm21" => :SSE21, "zmm21" => :SSE21,
    "xmm22" => :SSE22, "ymm22" => :SSE22, "zmm22" => :SSE22,
    "xmm23" => :SSE23, "ymm23" => :SSE23, "zmm23" => :SSE23,
    "xmm24" => :SSE24, "ymm24" => :SSE24, "zmm24" => :SSE24,
    "xmm25" => :SSE25, "ymm25" => :SSE25, "zmm25" => :SSE25,
    "xmm26" => :SSE26, "ymm26" => :SSE26, "zmm26" => :SSE26,
    "xmm27" => :SSE27, "ymm27" => :SSE27, "zmm27" => :SSE27,
    "xmm28" => :SSE28, "ymm28" => :SSE28, "zmm28" => :SSE28,
    "xmm29" => :SSE29, "ymm29" => :SSE29, "zmm29" => :SSE29,
    "xmm30" => :SSE30, "ymm30" => :SSE30, "zmm30" => :SSE30,
    "xmm31" => :SSE31, "ymm31" => :SSE31, "zmm31" => :SSE31,
)


############################################################## ASSEMBLY OPERANDS


abstract type AssemblyOperand end


struct AssemblyRegister <: AssemblyOperand
    name::String
end

Base.print(io::IO, reg::AssemblyRegister) = print(io, reg.name)


struct AssemblyMemoryOperand <: AssemblyOperand
    type::String
    address::String
end

Base.print(io::IO, mem::AssemblyMemoryOperand) =
    print(io, "*($(mem.address))")


struct AssemblyImmediate <: AssemblyOperand
    value::String
end

Base.print(io::IO, imm::AssemblyImmediate) = print(io, imm.value)


struct AssemblyOffset <: AssemblyOperand
    name::String
end

Base.print(io::IO, off::AssemblyOffset) = print(io, "offset ", off.name)


############################################################ ASSEMBLY STATEMENTS


abstract type AssemblyStatement end


struct AssemblyComment <: AssemblyStatement
    contents::String
end

Base.print(io::IO, comment::AssemblyComment) = print(io, comment.contents)


struct AssemblyLabel <: AssemblyStatement
    name::String
end

Base.print(io::IO, label::AssemblyLabel) = print(io, label.name, ':')


struct AssemblyInstruction <: AssemblyStatement
    opcode::String
    operands::Vector{AssemblyOperand}
    comment::String
end

AssemblyInstruction(instruction::AbstractString) =
    AssemblyInstruction(String(instruction), AssemblyOperand[], "")

AssemblyInstruction(instruction::AbstractString,
                    operands::Vector{T}) where {T <: AssemblyOperand} =
    AssemblyInstruction(String(instruction), operands, "")


const PRINT_HANDLERS = Dict{String,Function}()

function Base.print(io::IO, instr::AssemblyInstruction)
    if haskey(PRINT_HANDLERS, instr.opcode)
        PRINT_HANDLERS[instr.opcode](io, instr)
    else
        print(io, rpad('{' * instr.opcode * '}', 16))
        for (i, op) in enumerate(instr.operands)
            (i > 1) && print(io, ", ")
            print(io, op)
        end
        if !isempty(instr.comment)
            print(io, " // ", instr.comment)
        end
    end
    return nothing
end


############################################################### ASSEMBLY PARSING


function parse_assembly_operand(op::AbstractString)
    op = strip(op)
    if any(op == reg_name for reg_name in keys(X86_REGISTER_NAMES))
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
    # TODO: How to parse immediates?
    return AssemblyImmediate(op)
end


function parse_assembly_statement(stmt::AbstractString)

    # Return nothing for non-statements.
    if (stmt == "\t.text") || isempty(strip(stmt))
        return nothing
    end

    # Assembly comments begin with a semicolon.
    if startswith(strip(stmt), ';')
        return AssemblyComment(stmt)
    end

    # Tabs separate parts of an assembly statement.
    tokens = split(stmt, '\t')

    # If the statement contains no tabs, then it is a label.
    if length(tokens) == 1
        @assert endswith(tokens[1], ':')
        label_name = tokens[1][1:end-1]
        @assert !isempty(label_name)
        return AssemblyLabel(label_name)

    # If the statement begins with a tab, then it is an instruction.
    else
        @assert length(tokens) > 1
        @assert isempty(tokens[1])

        # If there is no second tab, then the instruction takes no operands.
        if length(tokens) == 2
            return AssemblyInstruction(tokens[2])

        # If there is a second tab, then the instruction operands follow it.
        else
            @assert length(tokens) == 3

            # Some instructions are output with a #-delimited comment.
            arg_tokens = split(tokens[3], " # ")
            if length(arg_tokens) == 1
                return AssemblyInstruction(tokens[2],
                    parse_assembly_operand.(split(arg_tokens[1], ',')))
            else
                @assert length(arg_tokens) == 2
                return AssemblyInstruction(tokens[2],
                    parse_assembly_operand.(split(arg_tokens[1], ',')),
                    strip(arg_tokens[2]))
            end
        end
    end

    error("AssemblyView.jl INTERNAL ERROR: Unable to parse" *
          " assembly statement <<<$stmt>>>.")
end


function parsed_asm(@nospecialize(func), @nospecialize(types...);
                    keep_comments::Bool=false)::Vector{AssemblyStatement}

    # Support either a tuple of argument types or varargs.
    if (length(types) == 1) && !(types[1] isa Type)
        types = types[1]
    end

    # Call internal Julia API to generate x86 assembly code.
    code = _dump_function(func, types,
        true,   # Generate native code (as opposed to LLVM IR).
        false,  # Don't generate wrapper code.
        true,   # (strip_ir_metadata) Ignored when dumping native code.
        true,   # (dump_module) Ignored when dumping native code.
        :intel, # I prefer Intel assembly syntax.
        true,   # (optimize) Ignored when dumping native code.
        :source # TODO: What does debuginfo=:source mean?
    )

    # Parse each line of code, discarding comments if requested.
    result = AssemblyStatement[]
    for stmt in split(code, '\n')
        parsed_stmt = parse_assembly_statement(stmt)
        if !isnothing(parsed_stmt) && (keep_comments ||
                                       !(parsed_stmt isa AssemblyComment))
            push!(result, parsed_stmt)
        end
    end

    return result
end


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


end # module AssemblyView
