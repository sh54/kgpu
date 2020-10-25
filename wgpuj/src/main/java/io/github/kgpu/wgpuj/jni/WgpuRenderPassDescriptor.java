package io.github.kgpu.wgpuj.jni;

import io.github.kgpu.wgpuj.WgpuJava;
import io.github.kgpu.wgpuj.util.WgpuJavaStruct;
import jnr.ffi.Runtime;
import jnr.ffi.Struct;

/** NOTE: THIS FILE WAS PRE-GENERATED BY JNR_GEN! */
public class WgpuRenderPassDescriptor extends WgpuJavaStruct {

    private final DynamicStructRef<WgpuColorAttachmentDescriptor> colorAttachments = new DynamicStructRef<>(WgpuColorAttachmentDescriptor.class);
    private final Struct.Unsigned64 colorAttachmentsLength = new Struct.Unsigned64();
    private final DynamicStructRef<WgpuDepthStencilAttachmentDescriptor> depthStencilAttachment = new DynamicStructRef<>(WgpuDepthStencilAttachmentDescriptor.class);

    protected WgpuRenderPassDescriptor(boolean direct){
         if(direct){
             useDirectMemory();
        }
    }

    @Deprecated
    public WgpuRenderPassDescriptor(Runtime runtime){
        super(runtime);
    }

    /**
    * Creates this struct on the java heap.
    * In general, this should <b>not</b> be used because these structs
    * cannot be directly passed into native code. 
    */
    public static WgpuRenderPassDescriptor createHeap(){
        return new WgpuRenderPassDescriptor(false);
    }

    /**
    * Creates this struct in direct memory.
    * This is how most structs should be created (unless, they
    * are members of a nothing struct)
    * 
    * @see WgpuJavaStruct#useDirectMemory
    */
    public static WgpuRenderPassDescriptor createDirect(){
        return new WgpuRenderPassDescriptor(true);
    }


    public DynamicStructRef<WgpuColorAttachmentDescriptor> getColorAttachments(){
        return colorAttachments;
    }

    public void setColorAttachments(WgpuColorAttachmentDescriptor... x){
        if(x.length == 0 || x[0] == null){
            this.colorAttachments.set(WgpuJava.createNullPointer());
        } else {
            this.colorAttachments.set(x);
        }
    }

    public long getColorAttachmentsLength(){
        return colorAttachmentsLength.get();
    }

    public void setColorAttachmentsLength(long x){
        this.colorAttachmentsLength.set(x);
    }

    public DynamicStructRef<WgpuDepthStencilAttachmentDescriptor> getDepthStencilAttachment(){
        return depthStencilAttachment;
    }

    public void setDepthStencilAttachment(WgpuDepthStencilAttachmentDescriptor... x){
        if(x.length == 0 || x[0] == null){
            this.depthStencilAttachment.set(WgpuJava.createNullPointer());
        } else {
            this.depthStencilAttachment.set(x);
        }
    }

}