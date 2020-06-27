#include "cuda_renderer/model.h"
#include <assert.h>

using namespace cuda_renderer;

cuda_renderer::Model::~Model()
{

}

cuda_renderer::Model::Model(const std::string &fileName)
{
    LoadModel(fileName);
}

void cuda_renderer::Model::LoadModel(const std::string &fileName)
{
    scene = aiImportFile(fileName.c_str(), aiProcessPreset_TargetRealtime_Quality);

    {
        tris.clear();
        size_t guess_size = scene->mMeshes[scene->mRootNode->mMeshes[0]]->mNumFaces;
        tris.reserve(guess_size);
    }
    {
        faces.clear();
        size_t guess_size = scene->mMeshes[scene->mRootNode->mMeshes[0]]->mNumFaces;
        faces.reserve(guess_size);
    }
    {
        vertices.clear();
        size_t guess_size = scene->mMeshes[scene->mRootNode->mMeshes[0]]->mNumVertices;
        vertices.reserve(guess_size);
    }
    recursive_render(scene, scene->mRootNode);

    get_bounding_box(bbox_min, bbox_max);

    aiReleaseImport(scene);

    std::cout << "load model success    " <<                 std::endl;
    std::cout << "face(triangles) nums: " << faces.size() << std::endl;
    std::cout << "       vertices nums: " <<vertices.size()<<std::endl;

    if(faces.size() > 10000)
        std::cout << "you may want tools like meshlab to simplify models to speed up rendering" << std::endl;

    std::cout << "------------------------------------\n" << std::endl;
}

cuda_renderer::Model::float3 cuda_renderer::Model::mat_mul_vec(const aiMatrix4x4 &mat, const aiVector3D &vec)
{
    return {
        mat.a1*vec.x + mat.a2*vec.y + mat.a3*vec.z + mat.a4,
        mat.b1*vec.x + mat.b2*vec.y + mat.b3*vec.z + mat.b4,
        mat.c1*vec.x + mat.c2*vec.y + mat.c3*vec.z + mat.c4,
    };
}

void cuda_renderer::Model::recursive_render(const aiScene *sc, const aiNode *nd, aiMatrix4x4 m)
{
    aiMultiplyMatrix4(&m, &nd->mTransformation);

    printf("Reading faces\n");
    for (size_t n=0; n < nd->mNumMeshes; ++n){
        const struct aiMesh* mesh = sc->mMeshes[nd->mMeshes[n]];

        for (size_t t = 0; t < mesh->mNumFaces; ++t){
            const struct aiFace* face = &mesh->mFaces[t];
            // std::cout << face->mNumIndices << std::endl;
            if (face->mNumIndices < 3) continue;
            assert(face->mNumIndices == 3 && "we only render triangle, use tools like meshlab to modify this models");

            Triangle tri_temp;
            tri_temp.v0 = mat_mul_vec(m, mesh->mVertices[face->mIndices[0]]);
            tri_temp.v1 = mat_mul_vec(m, mesh->mVertices[face->mIndices[1]]);
            tri_temp.v2 = mat_mul_vec(m, mesh->mVertices[face->mIndices[2]]);
            int r,g,b;
            if ( mesh->mColors[0] != NULL)
            {
                aiColor4D color = mesh->mColors[0][face->mIndices[0]];
                r = round(color.r*255);
                g = round(color.g*255);
                b = round(color.b*255);
            }
            else 
            {
                r = 128;
                g = 128;
                b = 128;
            }
            tri_temp.color.v0 = r;
            tri_temp.color.v1 = g;
            tri_temp.color.v2 = b;
            // aiColor4D color1 = mesh->mColors[0][face->mIndices[1]];
            // aiColor4D color2 = mesh->mColors[0][face->mIndices[2]];
            // float r = color.r;
            // float r1 = color1.r;
            // float r2 = color2.r;
            // int ri = round(r*255);
            // int ri1 = round(r1*255);
            // int ri2 = round(r2*255);
            // if(n==0 && t==0){
            //     std::cout<< tri_temp.color << std::endl;
            //     int red = (tri_temp.color >> 16) & 0xFF;
            //     int green = (tri_temp.color >> 8) & 0xFF;
            //     int blue = tri_temp.color & 0xFF;
            //     std::cout<< red << std::endl;
            //     std::cout<< green << std::endl;
            //     std::cout<< blue << std::endl;
                
            // }
            tris.push_back(tri_temp);

            int3 face_temp;
            face_temp.v0 = face->mIndices[0];
            face_temp.v1 = face->mIndices[1];
            face_temp.v2 = face->mIndices[2];
            faces.push_back(face_temp);
        }

        for(size_t t = 0; t < mesh->mNumVertices; ++t){
            float3 v;
            v.x = mesh->mVertices[t].x;
            v.y = mesh->mVertices[t].y;
            v.z = mesh->mVertices[t].z;
            vertices.push_back(v);
        }
    }

    // draw all children
    printf("Reading children faces\n");
    for (size_t n = 0; n < nd->mNumChildren; ++n)
        recursive_render(sc, nd->mChildren[n], m);
}

void cuda_renderer::Model::get_bounding_box_for_node(const aiNode *nd, aiVector3D& min, aiVector3D& max, aiMatrix4x4 *trafo) const
{
    aiMatrix4x4 prev; // Use struct keyword to show you want struct version of this, not normal typedef?
    unsigned int n = 0, t;

    prev = *trafo;
    aiMultiplyMatrix4(trafo, &nd->mTransformation);

    for (; n < nd->mNumMeshes; ++n)
    {
      const struct aiMesh* mesh = scene->mMeshes[nd->mMeshes[n]];
      for (t = 0; t < mesh->mNumVertices; ++t)
      {
        aiVector3D tmp = mesh->mVertices[t];
        aiTransformVecByMatrix4(&tmp, trafo);

        min.x = std::min(min.x,tmp.x);
        min.y = std::min(min.y,tmp.y);
        min.z = std::min(min.z,tmp.z);

        max.x = std::max(max.x,tmp.x);
        max.y = std::max(max.y,tmp.y);
        max.z = std::max(max.z,tmp.z);
      }
    }

    for (n = 0; n < nd->mNumChildren; ++n)
      get_bounding_box_for_node(nd->mChildren[n], min, max, trafo);

    *trafo = prev;
}

void cuda_renderer::Model::get_bounding_box(aiVector3D& min, aiVector3D &max) const
{
    aiMatrix4x4 trafo;
    aiIdentityMatrix4(&trafo);

    min.x = min.y = min.z = 1e10f;
    max.x = max.y = max.z = -1e10f;
    get_bounding_box_for_node(scene->mRootNode, min, max, &trafo);
}