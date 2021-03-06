package io.github.kgpu.wgpuj.jni;

import io.github.kgpu.wgpuj.util.WgpuJavaStruct;
import jnr.ffi.Runtime;
import jnr.ffi.Struct;

/** NOTE: THIS FILE WAS PRE-GENERATED BY JNR_GEN! */
public class WgpuColorStateDescriptor extends WgpuJavaStruct {

    private final Struct.Enum<WgpuTextureFormat> format =
            new Struct.Enum<>(WgpuTextureFormat.class);
    private final WgpuBlendDescriptor alphaBlend = inner(WgpuBlendDescriptor.createHeap());
    private final WgpuBlendDescriptor colorBlend = inner(WgpuBlendDescriptor.createHeap());
    private final Struct.Unsigned32 writeMask = new Struct.Unsigned32();

    protected WgpuColorStateDescriptor(boolean direct) {
        if (direct) {
            useDirectMemory();
        }
    }

    @Deprecated
    public WgpuColorStateDescriptor(Runtime runtime) {
        super(runtime);
    }

    /**
     * Creates this struct on the java heap. In general, this should <b>not</b> be used because
     * these structs cannot be directly passed into native code.
     */
    public static WgpuColorStateDescriptor createHeap() {
        return new WgpuColorStateDescriptor(false);
    }

    /**
     * Creates this struct in direct memory. This is how most structs should be created (unless,
     * they are members of a nothing struct)
     *
     * @see WgpuJavaStruct#useDirectMemory
     */
    public static WgpuColorStateDescriptor createDirect() {
        return new WgpuColorStateDescriptor(true);
    }

    public WgpuTextureFormat getFormat() {
        return format.get();
    }

    public void setFormat(WgpuTextureFormat x) {
        this.format.set(x);
    }

    public WgpuBlendDescriptor getAlphaBlend() {
        return alphaBlend;
    }

    public WgpuBlendDescriptor getColorBlend() {
        return colorBlend;
    }

    public long getWriteMask() {
        return writeMask.get();
    }

    public void setWriteMask(long x) {
        this.writeMask.set(x);
    }
}
