package io.github.kgpu.wgpuj.jni;

import io.github.kgpu.wgpuj.util.CStrPointer;
import io.github.kgpu.wgpuj.util.RustCString;
import io.github.kgpu.wgpuj.util.WgpuJavaStruct;
import jnr.ffi.Runtime;
import jnr.ffi.Struct;

/** NOTE: THIS FILE WAS PRE-GENERATED BY JNR_GEN! */
public class WgpuCAdapterInfo extends WgpuJavaStruct {

    private final @CStrPointer Struct.Pointer name = new Struct.Pointer();
    private final Struct.Unsigned64 nameLength = new Struct.Unsigned64();
    private final Struct.Unsigned64 vendor = new Struct.Unsigned64();
    private final Struct.Unsigned64 device = new Struct.Unsigned64();
    private final Struct.Enum<WgpuCDeviceType> deviceType =
            new Struct.Enum<>(WgpuCDeviceType.class);
    private final Struct.Enum<WgpuBackend> backend = new Struct.Enum<>(WgpuBackend.class);

    protected WgpuCAdapterInfo(boolean direct) {
        if (direct) {
            useDirectMemory();
        }
    }

    @Deprecated
    public WgpuCAdapterInfo(Runtime runtime) {
        super(runtime);
    }

    /**
     * Creates this struct on the java heap. In general, this should <b>not</b> be used because
     * these structs cannot be directly passed into native code.
     */
    public static WgpuCAdapterInfo createHeap() {
        return new WgpuCAdapterInfo(false);
    }

    /**
     * Creates this struct in direct memory. This is how most structs should be created (unless,
     * they are members of a nothing struct)
     *
     * @see WgpuJavaStruct#useDirectMemory
     */
    public static WgpuCAdapterInfo createDirect() {
        return new WgpuCAdapterInfo(true);
    }

    public java.lang.String getName() {
        return RustCString.fromPointer(name.get());
    }

    public void setName(java.lang.String x) {
        this.name.set(RustCString.toPointer(x));
    }

    public long getNameLength() {
        return nameLength.get();
    }

    public void setNameLength(long x) {
        this.nameLength.set(x);
    }

    public long getVendor() {
        return vendor.get();
    }

    public void setVendor(long x) {
        this.vendor.set(x);
    }

    public long getDevice() {
        return device.get();
    }

    public void setDevice(long x) {
        this.device.set(x);
    }

    public WgpuCDeviceType getDeviceType() {
        return deviceType.get();
    }

    public void setDeviceType(WgpuCDeviceType x) {
        this.deviceType.set(x);
    }

    public WgpuBackend getBackend() {
        return backend.get();
    }

    public void setBackend(WgpuBackend x) {
        this.backend.set(x);
    }
}
