package io.github.kgpu.wgpuj.jni;

import io.github.kgpu.wgpuj.util.WgpuJavaStruct;
import jnr.ffi.Runtime;
import jnr.ffi.Struct;

/** NOTE: THIS FILE WAS PRE-GENERATED BY JNR_GEN! */
public class WgpuPassChannel_Color extends WgpuJavaStruct {

    private final Struct.Enum<WgpuLoadOp> loadOp = new Struct.Enum<>(WgpuLoadOp.class);
    private final Struct.Enum<WgpuStoreOp> storeOp = new Struct.Enum<>(WgpuStoreOp.class);
    private final WgpuColor clearValue = inner(WgpuColor.createHeap());
    private final Struct.Boolean readOnly = new Struct.Boolean();

    protected WgpuPassChannel_Color(boolean direct){
         if(direct){
             useDirectMemory();
        }
    }

    @Deprecated
    public WgpuPassChannel_Color(Runtime runtime){
        super(runtime);
    }

    /**
    * Creates this struct on the java heap.
    * In general, this should <b>not</b> be used because these structs
    * cannot be directly passed into native code. 
    */
    public static WgpuPassChannel_Color createHeap(){
        return new WgpuPassChannel_Color(false);
    }

    /**
    * Creates this struct in direct memory.
    * This is how most structs should be created (unless, they
    * are members of a nothing struct)
    * 
    * @see WgpuJavaStruct#useDirectMemory
    */
    public static WgpuPassChannel_Color createDirect(){
        return new WgpuPassChannel_Color(true);
    }


    public WgpuLoadOp getLoadOp(){
        return loadOp.get();
    }

    public void setLoadOp(WgpuLoadOp x){
        this.loadOp.set(x);
    }

    public WgpuStoreOp getStoreOp(){
        return storeOp.get();
    }

    public void setStoreOp(WgpuStoreOp x){
        this.storeOp.set(x);
    }

    public WgpuColor getClearValue(){
        return clearValue;
    }

    public boolean getReadOnly(){
        return readOnly.get();
    }

    public void setReadOnly(boolean x){
        this.readOnly.set(x);
    }

}