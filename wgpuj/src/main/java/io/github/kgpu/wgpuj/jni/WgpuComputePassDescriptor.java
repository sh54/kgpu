package io.github.kgpu.wgpuj.jni;

import io.github.kgpu.wgpuj.util.WgpuJavaStruct;
import jnr.ffi.Runtime;
import jnr.ffi.Struct;

/** NOTE: THIS FILE WAS PRE-GENERATED BY JNR_GEN! */
public class WgpuComputePassDescriptor extends WgpuJavaStruct {

    private final Struct.Unsigned32 todo = new Struct.Unsigned32();

    protected WgpuComputePassDescriptor(boolean direct) {
        if (direct) {
            useDirectMemory();
        }
    }

    @Deprecated
    public WgpuComputePassDescriptor(Runtime runtime) {
        super(runtime);
    }

    /**
     * Creates this struct on the java heap. In general, this should <b>not</b> be used because
     * these structs cannot be directly passed into native code.
     */
    public static WgpuComputePassDescriptor createHeap() {
        return new WgpuComputePassDescriptor(false);
    }

    /**
     * Creates this struct in direct memory. This is how most structs should be created (unless,
     * they are members of a nothing struct)
     *
     * @see WgpuJavaStruct#useDirectMemory
     */
    public static WgpuComputePassDescriptor createDirect() {
        return new WgpuComputePassDescriptor(true);
    }

    public long getTodo() {
        return todo.get();
    }

    public void setTodo(long x) {
        this.todo.set(x);
    }
}
