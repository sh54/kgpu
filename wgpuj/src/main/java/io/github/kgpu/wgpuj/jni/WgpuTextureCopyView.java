package io.github.kgpu.wgpuj.jni;

import io.github.kgpu.wgpuj.util.WgpuJavaStruct;
import jnr.ffi.Runtime;
import jnr.ffi.Struct;

/** NOTE: THIS FILE WAS PRE-GENERATED BY JNR_GEN! */
public class WgpuTextureCopyView extends WgpuJavaStruct {

    private final Struct.Unsigned64 texture = new Struct.Unsigned64();
    private final Struct.Unsigned32 mipLevel = new Struct.Unsigned32();
    private final WgpuOrigin3d origin = inner(WgpuOrigin3d.createHeap());

    protected WgpuTextureCopyView(boolean direct) {
        if (direct) {
            useDirectMemory();
        }
    }

    @Deprecated
    public WgpuTextureCopyView(Runtime runtime) {
        super(runtime);
    }

    /**
     * Creates this struct on the java heap. In general, this should <b>not</b> be used because
     * these structs cannot be directly passed into native code.
     */
    public static WgpuTextureCopyView createHeap() {
        return new WgpuTextureCopyView(false);
    }

    /**
     * Creates this struct in direct memory. This is how most structs should be created (unless,
     * they are members of a nothing struct)
     *
     * @see WgpuJavaStruct#useDirectMemory
     */
    public static WgpuTextureCopyView createDirect() {
        return new WgpuTextureCopyView(true);
    }

    public long getTexture() {
        return texture.get();
    }

    public void setTexture(long x) {
        this.texture.set(x);
    }

    public long getMipLevel() {
        return mipLevel.get();
    }

    public void setMipLevel(long x) {
        this.mipLevel.set(x);
    }

    public WgpuOrigin3d getOrigin() {
        return origin;
    }
}
