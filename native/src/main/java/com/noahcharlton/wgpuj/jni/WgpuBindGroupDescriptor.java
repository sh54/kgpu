package com.noahcharlton.wgpuj.jni;

import com.noahcharlton.wgpuj.WgpuJava;
import com.noahcharlton.wgpuj.util.WgpuJavaStruct;
import com.noahcharlton.wgpuj.util.CStrPointer;
import com.noahcharlton.wgpuj.util.RustCString;
import jnr.ffi.Runtime;
import jnr.ffi.Struct;

/** NOTE: THIS FILE WAS PRE-GENERATED BY JNR_GEN! */
public class WgpuBindGroupDescriptor extends WgpuJavaStruct {

    private final @CStrPointer Struct.Pointer label = new Struct.Pointer();
    private final Struct.Unsigned64 layout = new Struct.Unsigned64();
    private final DynamicStructRef<WgpuBindGroupEntry> entries = new DynamicStructRef<>(WgpuBindGroupEntry.class);
    private final Struct.Unsigned64 entriesLength = new Struct.Unsigned64();

    private WgpuBindGroupDescriptor(){}

    @Deprecated
    public WgpuBindGroupDescriptor(Runtime runtime){
        super(runtime);
    }

    /**
    * Creates this struct on the java heap.
    * In general, this should <b>not</b> be used because these structs
    * cannot be directly passed into native code. 
    */
    public static WgpuBindGroupDescriptor createHeap(){
        return new WgpuBindGroupDescriptor();
    }

    /**
    * Creates this struct in direct memory.
    * This is how most structs should be created (unless, they
    * are members of a nothing struct)
    * 
    * @see WgpuJavaStruct#useDirectMemory
    */
    public static WgpuBindGroupDescriptor createDirect(){
        var struct = new WgpuBindGroupDescriptor();
        struct.useDirectMemory();
        return struct;
    }


    public java.lang.String getLabel(){
        return RustCString.fromPointer(label.get());
    }

    public void setLabel(java.lang.String x){
        this.label.set(RustCString.toPointer(x));
    }

    public long getLayout(){
        return layout.get();
    }

    public void setLayout(long x){
        this.layout.set(x);
    }

    public DynamicStructRef<WgpuBindGroupEntry> getEntries(){
        return entries;
    }

    public void setEntries(WgpuBindGroupEntry... x){
        if(x.length == 0 || x[0] == null){
            this.entries.set(WgpuJava.createNullPointer());
        } else {
            this.entries.set(x);
        }
    }

    public long getEntriesLength(){
        return entriesLength.get();
    }

    public void setEntriesLength(long x){
        this.entriesLength.set(x);
    }

}