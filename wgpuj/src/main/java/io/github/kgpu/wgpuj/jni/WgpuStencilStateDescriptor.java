package io.github.kgpu.wgpuj.jni;

import io.github.kgpu.wgpuj.util.WgpuJavaStruct;
import jnr.ffi.Runtime;
import jnr.ffi.Struct;

/** NOTE: THIS FILE WAS PRE-GENERATED BY JNR_GEN! */
public class WgpuStencilStateDescriptor extends WgpuJavaStruct {

    private final WgpuStencilStateFaceDescriptor front =
            inner(WgpuStencilStateFaceDescriptor.createHeap());
    private final WgpuStencilStateFaceDescriptor back =
            inner(WgpuStencilStateFaceDescriptor.createHeap());
    private final Struct.Unsigned32 readMask = new Struct.Unsigned32();
    private final Struct.Unsigned32 writeMask = new Struct.Unsigned32();

    protected WgpuStencilStateDescriptor(boolean direct) {
        if (direct) {
            useDirectMemory();
        }
    }

    @Deprecated
    public WgpuStencilStateDescriptor(Runtime runtime) {
        super(runtime);
    }

    /**
     * Creates this struct on the java heap. In general, this should <b>not</b> be used because
     * these structs cannot be directly passed into native code.
     */
    public static WgpuStencilStateDescriptor createHeap() {
        return new WgpuStencilStateDescriptor(false);
    }

    /**
     * Creates this struct in direct memory. This is how most structs should be created (unless,
     * they are members of a nothing struct)
     *
     * @see WgpuJavaStruct#useDirectMemory
     */
    public static WgpuStencilStateDescriptor createDirect() {
        return new WgpuStencilStateDescriptor(true);
    }

    public WgpuStencilStateFaceDescriptor getFront() {
        return front;
    }

    public WgpuStencilStateFaceDescriptor getBack() {
        return back;
    }

    public long getReadMask() {
        return readMask.get();
    }

    public void setReadMask(long x) {
        this.readMask.set(x);
    }

    public long getWriteMask() {
        return writeMask.get();
    }

    public void setWriteMask(long x) {
        this.writeMask.set(x);
    }
}
