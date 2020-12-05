package io.github.kgpu.wgpuj.jni;

import io.github.kgpu.wgpuj.util.WgpuJavaStruct;
import jnr.ffi.Runtime;
import jnr.ffi.Struct;

/** NOTE: THIS FILE WAS PRE-GENERATED BY JNR_GEN! */
public class WgpuRequestAdapterOptions extends WgpuJavaStruct {

    private final Struct.Enum<WgpuPowerPreference> powerPreference =
            new Struct.Enum<>(WgpuPowerPreference.class);
    private final Struct.Unsigned64 compatibleSurface = new Struct.Unsigned64();

    protected WgpuRequestAdapterOptions(boolean direct) {
        if (direct) {
            useDirectMemory();
        }
    }

    @Deprecated
    public WgpuRequestAdapterOptions(Runtime runtime) {
        super(runtime);
    }

    /**
     * Creates this struct on the java heap. In general, this should <b>not</b> be used because
     * these structs cannot be directly passed into native code.
     */
    public static WgpuRequestAdapterOptions createHeap() {
        return new WgpuRequestAdapterOptions(false);
    }

    /**
     * Creates this struct in direct memory. This is how most structs should be created (unless,
     * they are members of a nothing struct)
     *
     * @see WgpuJavaStruct#useDirectMemory
     */
    public static WgpuRequestAdapterOptions createDirect() {
        return new WgpuRequestAdapterOptions(true);
    }

    public WgpuPowerPreference getPowerPreference() {
        return powerPreference.get();
    }

    public void setPowerPreference(WgpuPowerPreference x) {
        this.powerPreference.set(x);
    }

    public long getCompatibleSurface() {
        return compatibleSurface.get();
    }

    public void setCompatibleSurface(long x) {
        this.compatibleSurface.set(x);
    }
}
