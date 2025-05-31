# É necessario mudar vcarias linhas ainda.. 

from fastapi import FastAPI, Query, HTTPException
from typing import List
from kdtree_wrapper import lib, Tarv, TReg, EMBEDDING_DIM, ID_LEN # Import constants
from pydantic import BaseModel, Field
import ctypes

app = FastAPI()



class EmbeddingInput(BaseModel):
    embedding: List[float] = Field(..., min_length=EMBEDDING_DIM, max_length=EMBEDDING_DIM)
    id: str = Field(..., max_length=ID_LEN -1)
class EmbeddingOutput(BaseModel):
    embedding: List[float]
    id: str

tree_is_built_by_api = False


@app.post("/construir-arvore", summary="Initialize or Re-initialize the KD-Tree")
def constroi_arvore_endpoint():
    global tree_is_built_by_api
    try:
        lib.kdtree_construir()
        tree_is_built_by_api = True
        return {"mensagem": f"Árvore KD (dim={EMBEDDING_DIM}) inicializada/reconstruída com sucesso."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Falha ao construir árvore KD: {str(e)}")


@app.post("/inserir", summary="Insert a face embedding and ID into the KD-Tree")
def inserir_endpoint(data: EmbeddingInput):
    global tree_is_built_by_api
    if not tree_is_built_by_api:
  
        print("Árvore não construída via endpoint, construindo agora antes de inserir...")
        lib.kdtree_construir()
        tree_is_built_by_api = True
   
    try:

        c_embedding_array = (ctypes.c_float * EMBEDDING_DIM)(*data.embedding)
        id_bytes = data.id.encode('utf-8')[:ID_LEN-1]

        novo_ponto_treg = TReg(embedding=c_embedding_array, id=id_bytes)

        lib.inserir_ponto(novo_ponto_treg)

        return {"mensagem": f"Embedding para ID '{data.id}' inserido com sucesso."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao inserir ponto: {str(e)}")


@app.post("/buscar", response_model=EmbeddingOutput, summary="Find the closest face embedding")
def buscar_endpoint(query_data: EmbeddingInput): 
    global tree_is_built_by_api
    if not tree_is_built_by_api:
       
        print("Árvore não construída via endpoint, construindo agora antes de buscar...")
        lib.kdtree_construir()
        tree_is_built_by_api = True
        

    try:
      
        arv_ptr = lib.get_tree()
        if not arv_ptr or not arv_ptr.contents.raiz:
             
            if not arv_ptr:
                 raise HTTPException(status_code=500, detail="Falha ao obter ponteiro da árvore KD.")
            if not arv_ptr.contents.raiz:
                 raise HTTPException(status_code=404, detail="Árvore KD está vazia. Insira pontos antes de buscar.")


       
        c_query_embedding_array = (ctypes.c_float * EMBEDDING_DIM)(*query_data.embedding)
       
        query_id_bytes = query_data.id.encode('utf-8')[:ID_LEN-1]
        query_treg = TReg(embedding=c_query_embedding_array, id=query_id_bytes)

    
        resultado_treg = lib.buscar_mais_proximo(arv_ptr, query_treg)

       
        result_id = resultado_treg.id.decode('utf-8', errors='ignore')
        if result_id == "NOT_FOUND" or not any(resultado_treg.embedding): 
             raise HTTPException(status_code=404, detail="Nenhum ponto correspondente encontrado na árvore KD.")

        result_embedding = list(resultado_treg.embedding)


        return EmbeddingOutput(embedding=result_embedding, id=result_id)

    except HTTPException as e: 
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro durante a busca: {str(e)}")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
