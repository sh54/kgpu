package io.github.kgpu.wgpuj.jni;

import io.github.kgpu.wgpuj.WgpuJava;
import io.github.kgpu.wgpuj.util.WgpuJavaStruct;
import jnr.ffi.Runtime;
import jnr.ffi.Struct;

/** NOTE: THIS FILE WAS PRE-GENERATED BY JNR_GEN! */
public class WgpuVertexStateDescriptor extends WgpuJavaStruct {

    private final Struct.Enum<WgpuIndexFormat> indexFormat =
            new Struct.Enum<>(WgpuIndexFormat.class);
    private final DynamicStructRef<WgpuVertexBufferDescriptor> vertexBuffers =
            new DynamicStructRef<>(WgpuVertexBufferDescriptor.class);
    private final Struct.Unsigned64 vertexBuffersLength = new Struct.Unsigned64();

    protected WgpuVertexStateDescriptor(boolean direct) {
        if (direct) {
            useDirectMemory();
        }
    }

    @Deprecated
    public WgpuVertexStateDescriptor(Runtime runtime) {
        super(runtime);
    }

    /**
     * Creates this struct on the java heap. In general, this should <b>not</b> be used because
     * these structs cannot be directly passed into native code.
     */
    public static WgpuVertexStateDescriptor createHeap() {
        return new WgpuVertexStateDescriptor(false);
    }

    /**
     * Creates this struct in direct memory. This is how most structs should be created (unless,
     * they are members of a nothing struct)
     *
     * @see WgpuJavaStruct#useDirectMemory
     */
    public static WgpuVertexStateDescriptor createDirect() {
        return new WgpuVertexStateDescriptor(true);
    }

    public WgpuIndexFormat getIndexFormat() {
        return indexFormat.get();
    }

    public void setIndexFormat(WgpuIndexFormat x) {
        this.indexFormat.set(x);
    }

    public DynamicStructRef<WgpuVertexBufferDescriptor> getVertexBuffers() {
        return vertexBuffers;
    }

    public void setVertexBuffers(WgpuVertexBufferDescriptor... x) {
        if (x.length == 0 || x[0] == null) {
            this.vertexBuffers.set(WgpuJava.createNullPointer());
        } else {
            this.vertexBuffers.set(x);
        }
    }

    public long getVertexBuffersLength() {
        return vertexBuffersLength.get();
    }

    public void setVertexBuffersLength(long x) {
        this.vertexBuffersLength.set(x);
    }
}
