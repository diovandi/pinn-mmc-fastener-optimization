using Printf
using DelimitedFiles

# ============================================================================
# FreeFEM Wrapper for Cantilever Beam Validation
# ============================================================================

struct FreeFEMResults
    tip_deflection::Float64
    compliance::Float64
    max_stress::Float64
    support_pos::Float64
end

function run_freefem(support_pos::Float64; 
                     script_path::String="freefem_cantilever.edp",
                     freefem_path::String="FreeFem++")
    """
    Run FreeFEM script for cantilever beam with given support position.
    Returns FreeFEMResults struct with tip deflection, compliance, and max stress.
    """
    
    # Get directory of script
    script_dir = dirname(abspath(script_path))
    script_name = basename(script_path)
    
    # Create temporary script with support position
    temp_script = joinpath(script_dir, "freefem_temp.edp")
    
    # Read template script
    template_content = read(script_path, String)
    
    # Replace supportPos value in script
    # Find the line with "real supportPos = " and replace the value
    lines = split(template_content, '\n')
    modified_lines = []
    for line in lines
        if occursin(r"real supportPos\s*=", line)
            # Replace the value after = sign
            line = replace(line, r"=\s*[\d.]+" => "= $(support_pos)")
        end
        push!(modified_lines, line)
    end
    
    # Write temporary script
    write(temp_script, join(modified_lines, '\n'))
    
    # Run FreeFEM
    results_file = joinpath(script_dir, "freefem_results.txt")
    temp_script_name = basename(temp_script)
    cmd = `$freefem_path $temp_script_name`
    
    try
        cd(script_dir) do
            run(cmd)
        end
    catch e
        error("FreeFEM execution failed: $e")
    end
    
    # Parse results file
    if !isfile(results_file)
        error("FreeFEM results file not found: $results_file")
    end
    
    # Read results file
    results_dict = Dict{String, Float64}()
    for line in eachline(results_file)
        parts = split(strip(line))
        if length(parts) >= 2
            key = parts[1]
            val = parse(Float64, parts[2])
            results_dict[key] = val
        end
    end
    
    # Extract values
    tip_deflection = get(results_dict, "tip_deflection", NaN)
    compliance = get(results_dict, "compliance", NaN)
    max_stress = get(results_dict, "max_stress", NaN)
    support_pos_read = get(results_dict, "support_pos", support_pos)
    
    # Clean up temporary script
    if isfile(temp_script)
        rm(temp_script)
    end
    
    return FreeFEMResults(tip_deflection, compliance, max_stress, support_pos_read)
end

# Test function
function test()
    println("Testing FreeFEM wrapper...")
    results = run_freefem(0.5)
    println("Results:")
    @printf("  Tip deflection: %.6e m\n", results.tip_deflection)
    @printf("  Compliance: %.6e J\n", results.compliance)
    @printf("  Max stress: %.2f Pa\n", results.max_stress)
end

if abspath(PROGRAM_FILE) == @__FILE__
    test()
end

