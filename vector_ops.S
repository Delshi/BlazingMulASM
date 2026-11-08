/* Windows x64 ABI. LLVM integrated assembler, Intel syntax.
 * dot(a, b, n) -> float; had(a, b, dst, n) -> void
 * outer(a, n, b, m, dst) -> void
 * Non-overlapping float32 buffers. No C/C++ runtime or imported DLLs.
 * AVX kernels require usable AVX; dot_fma also requires FMA.
 */
.intel_syntax noprefix
.text

.p2align 4
.globl dot_scalar
dot_scalar:
    xorps xmm0, xmm0
    xor rax, rax
.Lds_loop:
    cmp rax, r8
    jae .Lds_done
    movss xmm1, DWORD PTR [rcx+rax*4]
    mulss xmm1, DWORD PTR [rdx+rax*4]
    addss xmm0, xmm1
    inc rax
    jmp .Lds_loop
.Lds_done:
    ret

.p2align 4
.globl dot_avx
dot_avx:
    vxorps ymm0, ymm0, ymm0
    mov r9, r8
    shr r9, 3
    je .Lda_reduce
.Lda_loop:
    vmovups ymm1, YMMWORD PTR [rcx]
    vmulps ymm1, ymm1, YMMWORD PTR [rdx]
    vaddps ymm0, ymm0, ymm1
    add rcx, 32
    add rdx, 32
    dec r9
    jne .Lda_loop
.Lda_reduce:
    vextractf128 xmm1, ymm0, 1
    vaddps xmm0, xmm0, xmm1
    vhaddps xmm0, xmm0, xmm0
    vhaddps xmm0, xmm0, xmm0
    and r8, 7
    je .Lda_end
.Lda_tail:
    vmovss xmm1, DWORD PTR [rcx]
    vmulss xmm1, xmm1, DWORD PTR [rdx]
    vaddss xmm0, xmm0, xmm1
    add rcx, 4
    add rdx, 4
    dec r8
    jne .Lda_tail
.Lda_end:
    vzeroupper
    ret

.p2align 4
.globl dot_fma
dot_fma:
    vxorps ymm0, ymm0, ymm0
    vxorps ymm1, ymm1, ymm1
    vxorps ymm2, ymm2, ymm2
    vxorps ymm3, ymm3, ymm3
    mov r9, r8
    shr r9, 5
    je .Ldf8
.Ldf32_loop:
    vmovups ymm4, YMMWORD PTR [rcx]
    vfmadd231ps ymm0, ymm4, YMMWORD PTR [rdx]
    vmovups ymm4, YMMWORD PTR [rcx+32]
    vfmadd231ps ymm1, ymm4, YMMWORD PTR [rdx+32]
    vmovups ymm4, YMMWORD PTR [rcx+64]
    vfmadd231ps ymm2, ymm4, YMMWORD PTR [rdx+64]
    vmovups ymm4, YMMWORD PTR [rcx+96]
    vfmadd231ps ymm3, ymm4, YMMWORD PTR [rdx+96]
    add rcx, 128
    add rdx, 128
    dec r9
    jne .Ldf32_loop
.Ldf8:
    and r8, 31
    mov r9, r8
    shr r9, 3
    je .Ldf_reduce
.Ldf8_loop:
    vmovups ymm4, YMMWORD PTR [rcx]
    vfmadd231ps ymm0, ymm4, YMMWORD PTR [rdx]
    add rcx, 32
    add rdx, 32
    dec r9
    jne .Ldf8_loop
.Ldf_reduce:
    vaddps ymm0, ymm0, ymm1
    vaddps ymm2, ymm2, ymm3
    vaddps ymm0, ymm0, ymm2
    vextractf128 xmm1, ymm0, 1
    vaddps xmm0, xmm0, xmm1
    vhaddps xmm0, xmm0, xmm0
    vhaddps xmm0, xmm0, xmm0
    and r8, 7
    je .Ldf_end
.Ldf_tail:
    vmovss xmm1, DWORD PTR [rcx]
    vfmadd231ss xmm0, xmm1, DWORD PTR [rdx]
    add rcx, 4
    add rdx, 4
    dec r8
    jne .Ldf_tail
.Ldf_end:
    vzeroupper
    ret

.p2align 4
.globl had_scalar
had_scalar:
    xor rax, rax
.Lhs_loop:
    cmp rax, r9
    jae .Lhs_end
    movss xmm0, DWORD PTR [rcx+rax*4]
    mulss xmm0, DWORD PTR [rdx+rax*4]
    movss DWORD PTR [r8+rax*4], xmm0
    inc rax
    jmp .Lhs_loop
.Lhs_end:
    ret

.p2align 4
.globl had_avx
had_avx:
    mov r10, r9
    shr r10, 3
    je .Lha_tail
.Lha_loop:
    vmovups ymm0, YMMWORD PTR [rcx]
    vmulps ymm0, ymm0, YMMWORD PTR [rdx]
    vmovups YMMWORD PTR [r8], ymm0
    add rcx, 32
    add rdx, 32
    add r8, 32
    dec r10
    jne .Lha_loop
.Lha_tail:
    and r9, 7
    je .Lha_end
.Lha_tail_loop:
    vmovss xmm0, DWORD PTR [rcx]
    vmulss xmm0, xmm0, DWORD PTR [rdx]
    vmovss DWORD PTR [r8], xmm0
    add rcx, 4
    add rdx, 4
    add r8, 4
    dec r9
    jne .Lha_tail_loop
.Lha_end:
    vzeroupper
    ret

.p2align 4
.globl had_unroll
had_unroll:
    mov r10, r9
    shr r10, 5
    je .Lhu8
.Lhu32_loop:
    vmovups ymm0, YMMWORD PTR [rcx]
    vmovups ymm1, YMMWORD PTR [rcx+32]
    vmovups ymm2, YMMWORD PTR [rcx+64]
    vmovups ymm3, YMMWORD PTR [rcx+96]
    vmulps ymm0, ymm0, YMMWORD PTR [rdx]
    vmulps ymm1, ymm1, YMMWORD PTR [rdx+32]
    vmulps ymm2, ymm2, YMMWORD PTR [rdx+64]
    vmulps ymm3, ymm3, YMMWORD PTR [rdx+96]
    vmovups YMMWORD PTR [r8], ymm0
    vmovups YMMWORD PTR [r8+32], ymm1
    vmovups YMMWORD PTR [r8+64], ymm2
    vmovups YMMWORD PTR [r8+96], ymm3
    add rcx, 128
    add rdx, 128
    add r8, 128
    dec r10
    jne .Lhu32_loop
.Lhu8:
    and r9, 31
    mov r10, r9
    shr r10, 3
    je .Lhu_tail
.Lhu8_loop:
    vmovups ymm0, YMMWORD PTR [rcx]
    vmulps ymm0, ymm0, YMMWORD PTR [rdx]
    vmovups YMMWORD PTR [r8], ymm0
    add rcx, 32
    add rdx, 32
    add r8, 32
    dec r10
    jne .Lhu8_loop
.Lhu_tail:
    and r9, 7
    je .Lhu_end
.Lhu_tail_loop:
    vmovss xmm0, DWORD PTR [rcx]
    vmulss xmm0, xmm0, DWORD PTR [rdx]
    vmovss DWORD PTR [r8], xmm0
    add rcx, 4
    add rdx, 4
    add r8, 4
    dec r9
    jne .Lhu_tail_loop
.Lhu_end:
    vzeroupper
    ret

.p2align 4
.globl outer_scalar
outer_scalar:
    test rdx, rdx
    jz .Lzero_outer_scalar
    test r9, r9
    jz .Lzero_outer_scalar
    mov r10, QWORD PTR [rsp+40]
.Los_rows:
    movss xmm0, DWORD PTR [rcx]
    xor rax, rax
.Los_cols:
    cmp rax, r9
    jae .Los_next
    movss xmm1, xmm0
    mulss xmm1, DWORD PTR [r8+rax*4]
    movss DWORD PTR [r10+rax*4], xmm1
    inc rax
    jmp .Los_cols
.Los_next:
    lea r10, [r10+r9*4]
    add rcx, 4
    dec rdx
    jne .Los_rows
.Los_end:
 .Lzero_outer_scalar:
    ret

.p2align 4
.globl outer_avx
outer_avx:
    test rdx, rdx
    jz .Lzero_outer_avx2
    test r9, r9
    jz .Lzero_outer_avx2
    mov r10, QWORD PTR [rsp+40]
.Loa_rows:
    vbroadcastss ymm0, DWORD PTR [rcx]
    xor rax, rax
    mov r11, r9
    shr r11, 3
    je .Loa_tail
.Loa_vec:
    vmovups ymm1, YMMWORD PTR [r8+rax*4]
    vmulps ymm1, ymm1, ymm0
    vmovups YMMWORD PTR [r10+rax*4], ymm1
    add rax, 8
    dec r11
    jne .Loa_vec
.Loa_tail:
    cmp rax, r9
    jae .Loa_next
.Loa_elem:
    vmovss xmm1, DWORD PTR [r8+rax*4]
    vmulss xmm1, xmm1, xmm0
    vmovss DWORD PTR [r10+rax*4], xmm1
    inc rax
    cmp rax, r9
    jb .Loa_elem
.Loa_next:
    lea r10, [r10+r9*4]
    add rcx, 4
    dec rdx
    jne .Loa_rows
.Loa_end:
    vzeroupper
 .Lzero_outer_avx2:
    ret

.p2align 4
.globl outer_unroll
outer_unroll:
    test rdx, rdx
    jz .Lzero_outer_avx2_unroll
    test r9, r9
    jz .Lzero_outer_avx2_unroll
    mov r10, QWORD PTR [rsp+40]
.Lou_rows:
    vbroadcastss ymm0, DWORD PTR [rcx]
    xor rax, rax
    mov r11, r9
    shr r11, 5
    je .Lou_vec8
.Lou_vec32:
    vmovups ymm1, YMMWORD PTR [r8+rax*4]
    vmovups ymm2, YMMWORD PTR [r8+rax*4+32]
    vmovups ymm3, YMMWORD PTR [r8+rax*4+64]
    vmovups ymm4, YMMWORD PTR [r8+rax*4+96]
    vmulps ymm1, ymm1, ymm0
    vmulps ymm2, ymm2, ymm0
    vmulps ymm3, ymm3, ymm0
    vmulps ymm4, ymm4, ymm0
    vmovups YMMWORD PTR [r10+rax*4], ymm1
    vmovups YMMWORD PTR [r10+rax*4+32], ymm2
    vmovups YMMWORD PTR [r10+rax*4+64], ymm3
    vmovups YMMWORD PTR [r10+rax*4+96], ymm4
    add rax, 32
    dec r11
    jne .Lou_vec32
.Lou_vec8:
    mov r11, r9
    sub r11, rax
    shr r11, 3
    je .Lou_tail
.Lou_vec8_loop:
    vmovups ymm1, YMMWORD PTR [r8+rax*4]
    vmulps ymm1, ymm1, ymm0
    vmovups YMMWORD PTR [r10+rax*4], ymm1
    add rax, 8
    dec r11
    jne .Lou_vec8_loop
.Lou_tail:
    cmp rax, r9
    jae .Lou_next
.Lou_elem:
    vmovss xmm1, DWORD PTR [r8+rax*4]
    vmulss xmm1, xmm1, xmm0
    vmovss DWORD PTR [r10+rax*4], xmm1
    inc rax
    cmp rax, r9
    jb .Lou_elem
.Lou_next:
    lea r10, [r10+r9*4]
    add rcx, 4
    dec rdx
    jne .Lou_rows
.Lou_end:
    vzeroupper
 .Lzero_outer_avx2_unroll:
    ret

/* Feature mask: bit0=AVX+OS state, bit1=FMA, bit2=AVX2. */
.p2align 4
.globl cpu_features
.seh_proc cpu_features
cpu_features:
    push rbx
    .seh_pushreg rbx
    .seh_endprologue
    xor r10d, r10d
    xor eax, eax
    cpuid
    mov r8d, eax
    cmp eax, 1
    jb .Lfeat_done
    mov eax, 1
    cpuid
    mov r9d, ecx
    and ecx, 0x1c000000
    cmp ecx, 0x1c000000
    jne .Lfeat_done
    xor ecx, ecx
    xgetbv
    and eax, 6
    cmp eax, 6
    jne .Lfeat_done
    or r10d, 1
    test r9d, 0x1000
    jz .Lfeat_avx2
    or r10d, 2
.Lfeat_avx2:
    cmp r8d, 7
    jb .Lfeat_done
    mov eax, 7
    xor ecx, ecx
    cpuid
    test ebx, 0x20
    jz .Lfeat_done
    or r10d, 4
.Lfeat_done:
    mov eax, r10d
    pop rbx
    ret
.seh_endproc

/* Batch: u64 kind, repeats; ptr a,b,dst; u64 n,m.
 * One Python->DLL call per series, not per kernel invocation.
 * kind 0..8: kernels below; 9..11: empty dot/had/outer controls.
 * Returns the last dot result (unspecified for array operations).
 */
.p2align 4
.globl run_batch
.seh_proc run_batch
run_batch:
    push rbx
    .seh_pushreg rbx
    push rsi
    .seh_pushreg rsi
    push rdi
    .seh_pushreg rdi
    sub rsp, 48
    .seh_stackalloc 48
    .seh_endprologue
    mov rbx, rcx
    mov rdi, QWORD PTR [rbx+8]
    xorps xmm0, xmm0
    test rdi, rdi
    jz .Lbatch_end
    mov rax, QWORD PTR [rbx]
    cmp rax, 0
    jne .Lselect_1
    lea rsi, [rip+dot_scalar]
    jmp .Lbatch_dot
.Lselect_1:
    cmp rax, 1
    jne .Lselect_2
    lea rsi, [rip+dot_avx]
    jmp .Lbatch_dot
.Lselect_2:
    cmp rax, 2
    jne .Lselect_3
    lea rsi, [rip+dot_fma]
    jmp .Lbatch_dot
.Lselect_3:
    cmp rax, 3
    jne .Lselect_4
    lea rsi, [rip+had_scalar]
    jmp .Lbatch_had
.Lselect_4:
    cmp rax, 4
    jne .Lselect_5
    lea rsi, [rip+had_avx]
    jmp .Lbatch_had
.Lselect_5:
    cmp rax, 5
    jne .Lselect_6
    lea rsi, [rip+had_unroll]
    jmp .Lbatch_had
.Lselect_6:
    cmp rax, 6
    jne .Lselect_7
    lea rsi, [rip+outer_scalar]
    jmp .Lbatch_outer
.Lselect_7:
    cmp rax, 7
    jne .Lselect_8
    lea rsi, [rip+outer_avx]
    jmp .Lbatch_outer
.Lselect_8:
    cmp rax, 8
    jne .Lselect_9
    lea rsi, [rip+outer_unroll]
    jmp .Lbatch_outer
.Lselect_9:
    cmp rax, 9
    jne .Lselect_10
    lea rsi, [rip+.Lempty]
    jmp .Lbatch_dot
.Lselect_10:
    cmp rax, 10
    jne .Lselect_11
    lea rsi, [rip+.Lempty]
    jmp .Lbatch_had
.Lselect_11:
    cmp rax, 11
    jne .Lselect_12
    lea rsi, [rip+.Lempty]
    jmp .Lbatch_outer
.Lselect_12:
    jmp .Lbatch_end
.Lbatch_dot:
    mov rcx, QWORD PTR [rbx+16]
    mov rdx, QWORD PTR [rbx+24]
    mov r8, QWORD PTR [rbx+40]
    call rsi
    dec rdi
    jnz .Lbatch_dot
    jmp .Lbatch_end
.Lbatch_had:
    mov rcx, QWORD PTR [rbx+16]
    mov rdx, QWORD PTR [rbx+24]
    mov r8, QWORD PTR [rbx+32]
    mov r9, QWORD PTR [rbx+40]
    call rsi
    dec rdi
    jnz .Lbatch_had
    jmp .Lbatch_end
.Lbatch_outer:
    mov rcx, QWORD PTR [rbx+16]
    mov rdx, QWORD PTR [rbx+40]
    mov r8, QWORD PTR [rbx+24]
    mov r9, QWORD PTR [rbx+48]
    mov rax, QWORD PTR [rbx+32]
    mov QWORD PTR [rsp+32], rax
    call rsi
    dec rdi
    jnz .Lbatch_outer
.Lbatch_end:
    sfence
    add rsp, 48
    pop rdi
    pop rsi
    pop rbx
    ret
.seh_endproc
.Lempty:
    ret
