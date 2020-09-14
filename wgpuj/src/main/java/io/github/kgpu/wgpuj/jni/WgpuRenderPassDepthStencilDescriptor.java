package io.github.kgpu.wgpuj.jni;

import io.github.kgpu.wgpuj.util.WgpuJavaStruct;
import jnr.ffi.Runtime;
import jnr.ffi.Struct;

/** NOTE: THIS FILE WAS PRE-GENERATED BY JNR_GEN! */
public class WgpuRenderPassDepthStencilDescriptor extends WgpuJavaStruct {

    private final Struct.Unsigned64 attachment = new Struct.Unsigned64();
    private final WgpuPassChannel_f32 depth = inner(WgpuPassChannel_f32.createHeap());
    private final WgpuPassChannel_u32 stencil = inner(WgpuPassChannel_u32.createHeap());

    protected WgpuRenderPassDepthStencilDescriptor(boolean direct){
         if(direct){
             useDirectMemory();
        }
    }

    @Deprecated
    public WgpuRenderPassDepthStencilDescriptor(Runtime runtime){
        super(runtime);
    }

    /**
    * Creates this struct on the java heap.
    * In general, this should <b>not</b> be used because these structs
    * cannot be directly passed into native code. 
    */
    public static WgpuRenderPassDepthStencilDescriptor createHeap(){
        return new WgpuRenderPassDepthStencilDescriptor(false);
    }

    /**
    * Creates this struct in direct memory.
    * This is how most structs should be created (unless, they
    * are members of a nothing struct)
    * 
    * @see WgpuJavaStruct#useDirectMemory
    */
    public static WgpuRenderPassDepthStencilDescriptor createDirect(){
        return new WgpuRenderPassDepthStencilDescriptor(true);
    }


    public long getAttachment(){
        return attachment.get();
    }

    public void setAttachment(long x){
        this.attachment.set(x);
    }

    public WgpuPassChannel_f32 getDepth(){
        return depth;
    }

    public WgpuPassChannel_u32 getStencil(){
        return stencil;
    }

}