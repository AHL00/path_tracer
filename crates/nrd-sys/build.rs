use std::env;
use std::fs;
use std::io;
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("cargo:rerun-if-changed=build.rs");

    let nrd_version = "4.14.3";

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    // Define paths
    let nrd_zip_path = out_dir.join(format!("nrd-v{}.zip", nrd_version));
    let nrd_src_dir = out_dir.join(format!("NRD-{}", nrd_version));
    let nrd_url = format!(
        "https://github.com/NVIDIA-RTX/NRD/archive/refs/tags/v{}.zip",
        nrd_version
    );

    // Download NRD source if not already present
    if !nrd_src_dir.exists() || !nrd_src_dir.join("CMakeLists.txt").exists() {
        println!(
            "cargo:warning=Downloading NRD v{} from {}",
            nrd_version, nrd_url
        );

        if let Some(parent) = nrd_zip_path.parent() {
            fs::create_dir_all(parent)?;
        }

        let response = ureq::get(&nrd_url).call()?;
        let mut body = response.into_body();
        let mut file = fs::File::create(&nrd_zip_path)?;
        io::copy(&mut body.as_reader(), &mut file)?;

        println!("cargo:warning=Extracting NRD to {}", nrd_src_dir.display());
        let file = fs::File::open(&nrd_zip_path)?;
        let mut archive = zip::ZipArchive::new(file)?;

        if !out_dir.exists() {
            fs::create_dir_all(&out_dir)?;
        }

        for i in 0..archive.len() {
            let mut file = archive.by_index(i)?;
            let outpath = out_dir.join(file.name());

            if file.name().ends_with('/') {
                fs::create_dir_all(&outpath)?;
            } else {
                if let Some(parent) = outpath.parent() {
                    if !parent.exists() {
                        fs::create_dir_all(parent)?;
                    }
                }
                let mut outfile = fs::File::create(&outpath)?;
                io::copy(&mut file, &mut outfile)?;

                #[cfg(unix)]
                {
                    use std::os::unix::fs::PermissionsExt;
                    if let Some(mode) = file.unix_mode() {
                        std::fs::set_permissions(&outpath, std::fs::Permissions::from_mode(mode))?;
                    }
                }
            }
        }
    }

    // Verify that we have the NRD source
    if !nrd_src_dir.exists() || !nrd_src_dir.join("CMakeLists.txt").exists() {
        return Err(format!(
            "NRD source directory '{}' not found or CMakeLists.txt missing after download and extraction", 
            nrd_src_dir.display()
        ).into());
    }

    println!("cargo:warning=Configuring and building NRD");

    let mut config = cmake::Config::new(&nrd_src_dir);

    config.define("NRD_STATIC_LIBRARY", "ON");
    config.define("NRD_DISABLE_SHADER_COMPILATION", "OFF");
    config.define("NRD_EMBEDS_SPIRV_SHADERS", "ON");

    // Platform-specific shader embedding options
    if cfg!(target_os = "windows") {
        config.define("NRD_EMBEDS_DXBC_SHADERS", "ON");
        config.define("NRD_EMBEDS_DXIL_SHADERS", "ON");

        config.define("CMAKE_CXX_FLAGS", "/EHsc");
        config.define("CMAKE_C_FLAGS", "/EHsc");
    } else {
        config.define("NRD_EMBEDS_DXBC_SHADERS", "OFF");
        config.define("NRD_EMBEDS_DXIL_SHADERS", "OFF");
    }

    config.profile("Release");

    if cfg!(target_os = "windows") {
        config.very_verbose(true).no_build_target(true);
    }

    let dst = config.build();

    // Manually copy the files since we're not using the install target
    if cfg!(target_os = "windows") {
        // Find the built library in the output directory
        let nrd_build_dir = dst.join("build");
        let lib_path = if nrd_build_dir.join("Release").exists() {
            nrd_build_dir.join("Release").join("NRD.lib")
        } else {
            nrd_build_dir.join("NRD.lib")
        };

        if !lib_path.exists() {
            // Try finding it in alternate locations
            let bin_dir = nrd_src_dir.join("_Bin");
            let alt_lib_path = if bin_dir.exists() {
                let try_path = bin_dir.join("Release").join("NRD.lib");
                if try_path.exists() {
                    try_path
                } else {
                    bin_dir.join("NRD.lib")
                }
            } else {
                return Err(format!("NRD library not found in the expected locations").into());
            };

            if alt_lib_path.exists() {
                println!(
                    "cargo:warning=Found NRD library at: {}",
                    alt_lib_path.display()
                );

                let lib_dir = dst.join("lib");
                fs::create_dir_all(&lib_dir)?;

                // Copy the library to the expected location
                fs::copy(alt_lib_path, lib_dir.join("NRD.lib"))?;
            } else {
                return Err(format!("Could not find the NRD library").into());
            }
        }
    }

    // Configure linking
    println!(
        "cargo:rustc-link-search=native={}",
        dst.join("lib").display()
    );

    if cfg!(windows) {
        println!("cargo:rustc-link-lib=static=NRD");
    } else {
        println!("cargo:rustc-link-lib=static=NRD");
    }

    // Link against platform-specific dependencies
    if cfg!(target_os = "windows") {
        if cfg!(target_env = "msvc") {
            // None
        }
    } else if cfg!(target_os = "linux") {
        println!("cargo:rustc-link-lib=stdc++");
    } else if cfg!(target_os = "macos") {
        println!("cargo:rustc-link-lib=c++");
    }

    Ok(())
}
