package io.github.kgpu.wgpuj.jni;

import io.github.kgpu.wgpuj.WgpuJava;
import io.github.kgpu.wgpuj.util.WgpuJavaStruct;
import io.github.kgpu.wgpuj.util.CStrPointer;
import io.github.kgpu.wgpuj.util.RustCString;
import jnr.ffi.Runtime;
import jnr.ffi.Struct;

/** NOTE: THIS FILE WAS PRE-GENERATED BY JNR_GEN! */
public class WgpuSamplerDescriptor extends WgpuJavaStruct {

    private final DynamicStructRef<WgpuChainedStruct> nextInChain = new DynamicStructRef<>(WgpuChainedStruct.class);
    private final @CStrPointer Struct.Pointer label = new Struct.Pointer();
    private final Struct.Enum<WgpuAddressMode> addressModeU = new Struct.Enum<>(WgpuAddressMode.class);
    private final Struct.Enum<WgpuAddressMode> addressModeV = new Struct.Enum<>(WgpuAddressMode.class);
    private final Struct.Enum<WgpuAddressMode> addressModeW = new Struct.Enum<>(WgpuAddressMode.class);
    private final Struct.Enum<WgpuFilterMode> magFilter = new Struct.Enum<>(WgpuFilterMode.class);
    private final Struct.Enum<WgpuFilterMode> minFilter = new Struct.Enum<>(WgpuFilterMode.class);
    private final Struct.Enum<WgpuFilterMode> mipmapFilter = new Struct.Enum<>(WgpuFilterMode.class);
    private final Struct.Float lodMinClamp = new Struct.Float();
    private final Struct.Float lodMaxClamp = new Struct.Float();
    private final Struct.Enum<WgpuCompareFunction> compare = new Struct.Enum<>(WgpuCompareFunction.class);
    private final Struct.Enum<WgpuSamplerBorderColor> borderColor = new Struct.Enum<>(WgpuSamplerBorderColor.class);

    protected WgpuSamplerDescriptor(boolean direct){
         if(direct){
             useDirectMemory();
        }
    }

    @Deprecated
    public WgpuSamplerDescriptor(Runtime runtime){
        super(runtime);
    }

    /**
    * Creates this struct on the java heap.
    * In general, this should <b>not</b> be used because these structs
    * cannot be directly passed into native code. 
    */
    public static WgpuSamplerDescriptor createHeap(){
        return new WgpuSamplerDescriptor(false);
    }

    /**
    * Creates this struct in direct memory.
    * This is how most structs should be created (unless, they
    * are members of a nothing struct)
    * 
    * @see WgpuJavaStruct#useDirectMemory
    */
    public static WgpuSamplerDescriptor createDirect(){
        return new WgpuSamplerDescriptor(true);
    }


    public DynamicStructRef<WgpuChainedStruct> getNextInChain(){
        return nextInChain;
    }

    public void setNextInChain(WgpuChainedStruct... x){
        if(x.length == 0 || x[0] == null){
            this.nextInChain.set(WgpuJava.createNullPointer());
        } else {
            this.nextInChain.set(x);
        }
    }

    public java.lang.String getLabel(){
        return RustCString.fromPointer(label.get());
    }

    public void setLabel(java.lang.String x){
        this.label.set(RustCString.toPointer(x));
    }

    public WgpuAddressMode getAddressModeU(){
        return addressModeU.get();
    }

    public void setAddressModeU(WgpuAddressMode x){
        this.addressModeU.set(x);
    }

    public WgpuAddressMode getAddressModeV(){
        return addressModeV.get();
    }

    public void setAddressModeV(WgpuAddressMode x){
        this.addressModeV.set(x);
    }

    public WgpuAddressMode getAddressModeW(){
        return addressModeW.get();
    }

    public void setAddressModeW(WgpuAddressMode x){
        this.addressModeW.set(x);
    }

    public WgpuFilterMode getMagFilter(){
        return magFilter.get();
    }

    public void setMagFilter(WgpuFilterMode x){
        this.magFilter.set(x);
    }

    public WgpuFilterMode getMinFilter(){
        return minFilter.get();
    }

    public void setMinFilter(WgpuFilterMode x){
        this.minFilter.set(x);
    }

    public WgpuFilterMode getMipmapFilter(){
        return mipmapFilter.get();
    }

    public void setMipmapFilter(WgpuFilterMode x){
        this.mipmapFilter.set(x);
    }

    public float getLodMinClamp(){
        return lodMinClamp.get();
    }

    public void setLodMinClamp(float x){
        this.lodMinClamp.set(x);
    }

    public float getLodMaxClamp(){
        return lodMaxClamp.get();
    }

    public void setLodMaxClamp(float x){
        this.lodMaxClamp.set(x);
    }

    public WgpuCompareFunction getCompare(){
        return compare.get();
    }

    public void setCompare(WgpuCompareFunction x){
        this.compare.set(x);
    }

    public WgpuSamplerBorderColor getBorderColor(){
        return borderColor.get();
    }

    public void setBorderColor(WgpuSamplerBorderColor x){
        this.borderColor.set(x);
    }

}