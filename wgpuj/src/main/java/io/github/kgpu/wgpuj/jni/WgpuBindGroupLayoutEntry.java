package io.github.kgpu.wgpuj.jni;

import io.github.kgpu.wgpuj.util.WgpuJavaStruct;
import jnr.ffi.Runtime;
import jnr.ffi.Struct;

/** NOTE: THIS FILE WAS PRE-GENERATED BY JNR_GEN! */
public class WgpuBindGroupLayoutEntry extends WgpuJavaStruct {

    private final Struct.Unsigned32 binding = new Struct.Unsigned32();
    private final Struct.Unsigned32 visibility = new Struct.Unsigned32();
    private final Struct.Enum<WgpuBindingType> ty = new Struct.Enum<>(WgpuBindingType.class);
    private final Struct.Boolean hasDynamicOffset = new Struct.Boolean();
    private final Struct.Unsigned64 minBufferBindingSize = new Struct.Unsigned64();
    private final Struct.Boolean multisampled = new Struct.Boolean();
    private final Struct.Enum<WgpuTextureViewDimension> viewDimension = new Struct.Enum<>(WgpuTextureViewDimension.class);
    private final Struct.Enum<WgpuTextureComponentType> textureComponentType = new Struct.Enum<>(WgpuTextureComponentType.class);
    private final Struct.Enum<WgpuTextureFormat> storageTextureFormat = new Struct.Enum<>(WgpuTextureFormat.class);
    private final Struct.Unsigned64 count = new Struct.Unsigned64();

    protected WgpuBindGroupLayoutEntry(boolean direct){
         if(direct){
             useDirectMemory();
        }
    }

    @Deprecated
    public WgpuBindGroupLayoutEntry(Runtime runtime){
        super(runtime);
    }

    /**
    * Creates this struct on the java heap.
    * In general, this should <b>not</b> be used because these structs
    * cannot be directly passed into native code. 
    */
    public static WgpuBindGroupLayoutEntry createHeap(){
        return new WgpuBindGroupLayoutEntry(false);
    }

    /**
    * Creates this struct in direct memory.
    * This is how most structs should be created (unless, they
    * are members of a nothing struct)
    * 
    * @see WgpuJavaStruct#useDirectMemory
    */
    public static WgpuBindGroupLayoutEntry createDirect(){
        return new WgpuBindGroupLayoutEntry(true);
    }


    public long getBinding(){
        return binding.get();
    }

    public void setBinding(long x){
        this.binding.set(x);
    }

    public long getVisibility(){
        return visibility.get();
    }

    public void setVisibility(long x){
        this.visibility.set(x);
    }

    public WgpuBindingType getTy(){
        return ty.get();
    }

    public void setTy(WgpuBindingType x){
        this.ty.set(x);
    }

    public boolean getHasDynamicOffset(){
        return hasDynamicOffset.get();
    }

    public void setHasDynamicOffset(boolean x){
        this.hasDynamicOffset.set(x);
    }

    public long getMinBufferBindingSize(){
        return minBufferBindingSize.get();
    }

    public void setMinBufferBindingSize(long x){
        this.minBufferBindingSize.set(x);
    }

    public boolean getMultisampled(){
        return multisampled.get();
    }

    public void setMultisampled(boolean x){
        this.multisampled.set(x);
    }

    public WgpuTextureViewDimension getViewDimension(){
        return viewDimension.get();
    }

    public void setViewDimension(WgpuTextureViewDimension x){
        this.viewDimension.set(x);
    }

    public WgpuTextureComponentType getTextureComponentType(){
        return textureComponentType.get();
    }

    public void setTextureComponentType(WgpuTextureComponentType x){
        this.textureComponentType.set(x);
    }

    public WgpuTextureFormat getStorageTextureFormat(){
        return storageTextureFormat.get();
    }

    public void setStorageTextureFormat(WgpuTextureFormat x){
        this.storageTextureFormat.set(x);
    }

    public long getCount(){
        return count.get();
    }

    public void setCount(long x){
        this.count.set(x);
    }

}