package io.github.kgpu.wgpuj.jnrgen;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Consumer;
import java.util.function.Predicate;

public class OutputHandler {

    private final File outputDirectory;
    private final String packageName = "io.github.kgpu.wgpuj.jni";
    private final String commentHeader = "/** NOTE: THIS FILE WAS PRE-GENERATED BY JNR_GEN! */";

    private final HashMap<String, Item> types = new HashMap<>();
    private final HashMap<String, String> aliases = new HashMap<>();
    private final HashMap<String, List<ConstantItem>> constants = new HashMap<>();
    private final Map<String, Consumer<Item>> hooks = Hooks.getHooks();

    private static final List<String> excluded = Arrays.asList(
            "WgpuBindingResource_WgpuBuffer_Body",
            "WgpuBindingResource_WgpuSampler_Body",
            "WgpuBindingResource_WgpuTextureView_Body",
            "WgpuRenderPassDepthStencilAttachmentDescriptorBase_TextureViewId",
            "WgpuRenderPassDescriptor",
            "WgpuRenderBundleEncoderDescriptor");

    private static final Map<String, String> exportNames = Map.of(
            "SURFACE_DESCRIPTOR_FROM_WINDOWS_H_W_N_D", "SURFACE_DESCRIPTOR_FROM_WINDOWS_HWND",
            "SURFACE_DESCRIPTOR_FROM_H_T_M_L_CANVAS_ID", "SURFACE_DESCRIPTOR_FROM_HTML_CANVAS_ID",
            "SHADER_MODULE_S_P_I_R_V_DESCRIPTOR", "SHADER_MODULE_SPIRV_DESCRIPTOR",
            "SHADER_MODULE_W_G_S_L_DESCRIPTOR", "SHADER_MODULE_WGSL_DESCRIPTOR",
            "WgpuRenderPassDepthStencilAttachmentDescriptorBase_TextureViewId", "WgpuRenderPassDepthStencilDescriptor",
            "WgpuRenderPassColorAttachmentDescriptorBase_TextureViewId", "WgpuRenderPassColorDescriptor"
    );

    public OutputHandler(File outputDirectory) {
        this.outputDirectory = outputDirectory;
    }

    public void saveConstants() throws IOException{
        var writer = startFile("Wgpu.java");
        writer.write("public final class Wgpu{\n\n");

        for(var entry : constants.entrySet()){
            saveConstantGroup(writer, entry);
        }

        writer.write("}");
        writer.flush();
        writer.close();
    }

    private void saveConstantGroup(BufferedWriter writer, Map.Entry<String, List<ConstantItem>> entry) throws IOException {
        var hasClass = !entry.getKey().isBlank();

        if(hasClass){
            writer.write("    public static final class ");
            writer.write(entry.getKey().replace("Wgpu", ""));
            writer.write("{\n");
        }

        for(ConstantItem constant : entry.getValue()) {
            constant.write(writer, hasClass ? "        " : "    ");
        }

        if(hasClass){
            writer.write("    }\n");
        }

        writer.write("\n");
    }

    public BufferedWriter startFile(String name, String... imports) throws IOException {
        File file = outputDirectory.toPath().resolve(name).toFile();

        var writer = new BufferedWriter(new FileWriter(file));
        writer.write("package ");
        writer.write(packageName);
        writer.write(";\n\n");

        for(String import_: imports){
            writer.write("import ");
            writer.write(import_);
            writer.write(";\n");
        }

        writer.write("\n");
        writer.write(commentHeader);
        writer.write("\n");

        return writer;
    }

    public void runHooks(Item item) {
        var hook = hooks.get(item.getJavaTypeName());

        if(hook != null)
            hook.accept(item);
    }

    public void registerType(String type, Item item){
        types.put(type, item);
    }

    public void registerTypeAlias(String actualType, String typeAlias){
        aliases.put(typeAlias, actualType);
    }

    public void registerConstant(String associatedType, ConstantItem item){
        if(constants.containsKey(associatedType)){
            constants.get(associatedType).add(item);
        }else{
            var list = new ArrayList<ConstantItem>();
            list.add(item);

            constants.put(associatedType, list);
        }
    }

    public static String toExportName(String key){
        return exportNames.getOrDefault(key, key);
    }

    public boolean containsType(String type) {
        return types.containsKey(type);
    }

    public boolean containsAlias(String type) {
        return aliases.containsKey(type);
    }

    public Item resolveType(String type) {
        return types.get(type);
    }

    public String getAlias(String type) {
        return aliases.get(type);
    }

    public static boolean isExcluded(String name) {
        return excluded.stream().anyMatch(Predicate.isEqual(name));
    }
}
