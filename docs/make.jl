using Documenter
# using LangevinDynamics

makedocs(
    sitename = "LangevinDynamics",
    format = Documenter.HTML()
    # ,
    # modules = [LangevinDynamics]
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
#=deploydocs(
    repo = "<repository url>"
)=#


deploydocs(;
    repo = "github.com/Yangyang-Tan/LangevinDynamics",
    devbranch = "main",
    devurl="dev",
    target = "build",
    branch = "gh-pages"
)
