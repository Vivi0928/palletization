from robosuite.utils.mjcf_utils import CustomMaterial

tex_attrib = { "type": "cube",}
mat_attrib = {
    "texrepeat": "1 1",
    "specular": "0.4",
    "shininess": "0.1",
}
redwood = CustomMaterial(
    texture="WoodRed",
    tex_name="redwood",
    mat_name="redwood_mat",
    tex_attrib=tex_attrib,
    mat_attrib=mat_attrib,
)
greenwood = CustomMaterial(
    texture="WoodGreen",
    tex_name="greenwood",
    mat_name="greenwood_mat",
    tex_attrib=tex_attrib,
    mat_attrib=mat_attrib,
)

lightwood = CustomMaterial(
    texture="WoodLight",
    tex_name="lightwood",
    mat_name="lightwood_mat",
    tex_attrib=tex_attrib,
    mat_attrib=mat_attrib,
)

darkwood = CustomMaterial(
    texture="WoodDark",
    tex_name="darkwood",
    mat_name="darkwood_mat",
    tex_attrib=tex_attrib,
    mat_attrib=mat_attrib,
)

tileswood = CustomMaterial(
    texture="WoodTiles",
    tex_name="tileswood",
    mat_name="tileswood_mat",
    tex_attrib=tex_attrib,
    mat_attrib=mat_attrib,
)