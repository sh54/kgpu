import org.jetbrains.dokka.gradle.DokkaTask

plugins {
    kotlin("multiplatform")
    id("maven-publish")
    id("org.jetbrains.dokka")
}

repositories {
    mavenCentral()
    jcenter()
}

group = rootProject.extra["projectGroup"]
version = rootProject.extra["projectVersion"]

kotlin {
    jvm()
    js().browser()

    sourceSets {
        val commonMain by getting {
            dependencies {
            }
        }
        val jvmMain by getting {
            dependencies {
                val lwjglVersion = rootProject.extra["lwjglVersion"]
                implementation("org.lwjgl:lwjgl-shaderc:$lwjglVersion")
                runtimeOnly("org.lwjgl:lwjgl-shaderc:$lwjglVersion:natives-windows")
                runtimeOnly("org.lwjgl:lwjgl-shaderc:$lwjglVersion:natives-macos")
                runtimeOnly("org.lwjgl:lwjgl-shaderc:$lwjglVersion:natives-linux")
            }
        }
        val jsMain by getting {
            dependencies {
                implementation("org.jetbrains.kotlinx:kotlinx-coroutines-core-js:1.3.7")
            }
        }
    }
}

tasks.withType<DokkaTask>().configureEach {
    outputDirectory = "$rootDir/docs/book/dokka/kshader"

    dokkaSourceSets {
        configureEach {
            includeNonPublic = false
        }

        register("commonMain") {
            displayName = "Common"
            platform = "common"
        }

        register("jvmMain") {
            displayName = "Desktop"
            platform = "jvm"
        }

        register("jsMain") {
            displayName = "Web"
            platform = "js"
        }
    }
}

publishing {
    repositories {
        maven {
            val releasesRepoUrl = "https://oss.sonatype.org/service/local/staging/deploy/maven2"
            val snapshotsRepoUrl = "https://oss.sonatype.org/content/repositories/snapshots"
            url = uri(if (version.toString().endsWith("SNAPSHOT")) snapshotsRepoUrl else releasesRepoUrl)

            credentials {
                username = "DevOrc"
                password = System.getenv("sonatypePassword") ?: ""
            }
        }
    }
}
