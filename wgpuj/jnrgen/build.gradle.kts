plugins {
    id("java")
}

val projectVersion: String by rootProject.extra
val projectGroup: String by rootProject.extra
group = projectGroup
version = projectVersion

java {
    sourceCompatibility = JavaVersion.VERSION_11
    targetCompatibility = JavaVersion.VERSION_11
}

repositories {
    mavenCentral()
}

tasks{
    register("deletePreviousBindings", Delete::class){
        delete("${buildDir}/jnr-gen")
    }
    
    register("generateBindings", JavaExec::class){
        dependsOn("deletePreviousBindings")
        dependsOn("classes")
        classpath = sourceSets["main"].runtimeClasspath
        main = "com.noahcharlton.wgpuj.jnrgen.JNRGenerator"
        args = listOf("${buildDir}")
    }
}