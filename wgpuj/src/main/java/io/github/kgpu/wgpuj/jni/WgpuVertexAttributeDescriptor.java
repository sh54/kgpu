package io.github.kgpu.wgpuj.jni;

import io.github.kgpu.wgpuj.util.WgpuJavaStruct;
import jnr.ffi.Runtime;
import jnr.ffi.Struct;

/** NOTE: THIS FILE WAS PRE-GENERATED BY JNR_GEN! */
public class WgpuVertexAttributeDescriptor extends WgpuJavaStruct {

    private final Struct.Unsigned64 offset = new Struct.Unsigned64();
    private final Struct.Enum<WgpuVertexFormat> format = new Struct.Enum<>(WgpuVertexFormat.class);
    private final Struct.Unsigned32 shaderLocation = new Struct.Unsigned32();

    protected WgpuVertexAttributeDescriptor(boolean direct) {
        if (direct) {
            useDirectMemory();
        }
    }

    @Deprecated
    public WgpuVertexAttributeDescriptor(Runtime runtime) {
        super(runtime);
    }

    /**
     * Creates this struct on the java heap. In general, this should <b>not</b> be used because
     * these structs cannot be directly passed into native code.
     */
    public static WgpuVertexAttributeDescriptor createHeap() {
        return new WgpuVertexAttributeDescriptor(false);
    }

    /**
     * Creates this struct in direct memory. This is how most structs should be created (unless,
     * they are members of a nothing struct)
     *
     * @see WgpuJavaStruct#useDirectMemory
     */
    public static WgpuVertexAttributeDescriptor createDirect() {
        return new WgpuVertexAttributeDescriptor(true);
    }

    public long getOffset() {
        return offset.get();
    }

    public void setOffset(long x) {
        this.offset.set(x);
    }

    public WgpuVertexFormat getFormat() {
        return format.get();
    }

    public void setFormat(WgpuVertexFormat x) {
        this.format.set(x);
    }

    public long getShaderLocation() {
        return shaderLocation.get();
    }

    public void setShaderLocation(long x) {
        this.shaderLocation.set(x);
    }
}
