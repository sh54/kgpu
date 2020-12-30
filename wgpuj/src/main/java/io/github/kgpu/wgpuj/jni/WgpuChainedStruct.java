package io.github.kgpu.wgpuj.jni;

import io.github.kgpu.wgpuj.WgpuJava;
import io.github.kgpu.wgpuj.util.WgpuJavaStruct;
import jnr.ffi.Runtime;
import jnr.ffi.Struct;

/** NOTE: THIS FILE WAS PRE-GENERATED BY JNR_GEN! */
public class WgpuChainedStruct extends WgpuJavaStruct {

    private final DynamicStructRef<WgpuChainedStruct> next =
            new DynamicStructRef<>(WgpuChainedStruct.class);
    private final Struct.Enum<WgpuSType> sType = new Struct.Enum<>(WgpuSType.class);

    protected WgpuChainedStruct(boolean direct) {
        if (direct) {
            useDirectMemory();
        }
    }

    @Deprecated
    public WgpuChainedStruct(Runtime runtime) {
        super(runtime);
    }

    /**
     * Creates this struct on the java heap. In general, this should <b>not</b> be used because
     * these structs cannot be directly passed into native code.
     */
    public static WgpuChainedStruct createHeap() {
        return new WgpuChainedStruct(false);
    }

    /**
     * Creates this struct in direct memory. This is how most structs should be created (unless,
     * they are members of a nothing struct)
     *
     * @see WgpuJavaStruct#useDirectMemory
     */
    public static WgpuChainedStruct createDirect() {
        return new WgpuChainedStruct(true);
    }

    public DynamicStructRef<WgpuChainedStruct> getNext() {
        return next;
    }

    public void setNext(WgpuChainedStruct... x) {
        if (x.length == 0 || x[0] == null) {
            this.next.set(WgpuJava.createNullPointer());
        } else {
            this.next.set(x);
        }
    }

    public WgpuSType getSType() {
        return sType.get();
    }

    public void setSType(WgpuSType x) {
        this.sType.set(x);
    }
}