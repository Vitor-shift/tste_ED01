#include<stdio.h>
#include<stdlib.h>
#include<float.h> // For DBL_MAX
#include<string.h>
#include<assert.h>
#include<math.h> // For fabs in distance, or for sqrt if actual distance needed

#define EMBEDDING_DIM 128
#define ID_LEN 100

/*Definições desenvolvedor usuario*/
typedef struct _reg{
    float embedding[EMBEDDING_DIM];
    char id[ID_LEN];
}treg;

void * aloca_reg(const float embedding[EMBEDDING_DIM], const char id[]){
    treg * reg = malloc(sizeof(treg));
    if (!reg) {
        perror("Failed to allocate treg");
        exit(EXIT_FAILURE);
    }
    memcpy(reg->embedding, embedding, EMBEDDING_DIM * sizeof(float));
    strncpy(reg->id, id, ID_LEN -1);
    reg->id[ID_LEN -1] = '\0'; 
    return reg;
}

// Comparador para floats
int comparador(void *a, void *b, int pos){
    treg *reg_a = (treg *)a;
    treg *reg_b = (treg *)b;

    if (pos < 0 || pos >= EMBEDDING_DIM) {
       
        fprintf(stderr, "Error: Invalid dimension access in comparador: %d\n", pos);
        return 0;
    }

    if (reg_a->embedding[pos] < reg_b->embedding[pos]){
        return -1; 
    } else if (reg_a->embedding[pos] > reg_b->embedding[pos]){
        return 1; 
    }
    return 0;
}

// Distância Euclidiana Quadrada
double distancia(void * a, void *b){
    treg *reg_a = (treg *)a;
    treg *reg_b = (treg *)b;
    double sum_sq_diff = 0.0;
    for (int i = 0; i < EMBEDDING_DIM; ++i){
        double diff = reg_a->embedding[i] - reg_b->embedding[i];
        sum_sq_diff += diff * diff;
    }
    return sum_sq_diff;
}


/*Definições desenvolvedor da biblioteca*/
typedef struct _node{
    void * key;
    struct _node * esq;
    struct _node * dir;
}tnode;

typedef struct _arv{
    tnode * raiz;
    int (*cmp)(void *, void *, int);
    double (*dist) (void *, void *);
    int k;
}tarv;


/*funções desenvolvedor da biblioteca*/

void kdtree_constroi_arv(tarv * arv, int (*cmp)(void *a, void *b, int ),double (*dist) (void *, void *),int k_dim){
    arv->raiz = NULL;
    arv->cmp = cmp;
    arv->dist = dist;
    arv->k = k_dim;
}

void _kdtree_insere(tnode **raiz, void * key, int (*cmp)(void *a, void *b, int),int profund, int k_dim){
    if(*raiz == NULL){
        *raiz = malloc(sizeof(tnode));
        if (!*raiz) {
             perror("Failed to allocate tnode for insertion");
             
             return;
        }
        (*raiz)->key = key; 
        (*raiz)->esq = NULL;
        (*raiz)->dir = NULL;
    }else{
        int pos = profund % k_dim;
       
        if (cmp( (*raiz)->key , key ,pos) < 0){ 
            _kdtree_insere( &((*raiz)->dir),key,cmp,profund + 1, k_dim);
        }else{
            _kdtree_insere( &((*raiz)->esq),key,cmp,profund + 1, k_dim);
        }
    }
}

void kdtree_insere(tarv *arv, void *key){
    _kdtree_insere(&(arv->raiz),key,arv->cmp,0,arv->k);
}


void _kdtree_destroi(tnode * node){
    if (node!=NULL){
        _kdtree_destroi(node->esq);
        _kdtree_destroi(node->dir);
        free(node->key);
        free(node);      
    }
}

void kdtree_destroi(tarv *arv){
    _kdtree_destroi(arv->raiz);
    arv->raiz = NULL;
}

void _kdtree_busca(tarv *arv, tnode ** atual_node_ptr, void * query_key, int profund, double *menor_dist, tnode **menor_node){
    tnode * atual_node = *atual_node_ptr;
    if (atual_node == NULL){
        return;
    }

    double dist_atual = arv->dist(atual_node->key, query_key);
    if (dist_atual < *menor_dist){
        *menor_dist = dist_atual;
        *menor_node = atual_node;
    }

    int pos = profund % arv->k;
    
    tnode ** lado_principal;
    tnode ** lado_oposto;


    int comparison_result = arv->cmp(query_key, atual_node->key, pos);

    if (comparison_result < 0){
        lado_principal = &(atual_node->esq);
        lado_oposto    = &(atual_node->dir);
    } else {
        lado_principal = &(atual_node->dir);
        lado_oposto    = &(atual_node->esq);
    }

    _kdtree_busca(arv, lado_principal, query_key, profund + 1, menor_dist, menor_node);

    float query_dim_val = ((treg*)query_key)->embedding[pos];
    float node_dim_val = ((treg*)atual_node->key)->embedding[pos];
    double diff_dim = query_dim_val - node_dim_val;
    double dist_sq_to_hyperplane = diff_dim * diff_dim;


    if (dist_sq_to_hyperplane < *menor_dist) {
        _kdtree_busca(arv, lado_oposto, query_key, profund + 1, menor_dist, menor_node);
    }
}


tnode * kdtree_busca_no(tarv *arv, void * key){
    tnode * menor = NULL;
    double menor_dist = DBL_MAX; 
    if (arv->raiz == NULL) return NULL; 
    _kdtree_busca(arv,&(arv->raiz),key,0,&menor_dist,&menor);
    return menor;
}


tarv arvore_global;
int arvore_inicializada = 0; 


void kdtree_construir() {
    if (arvore_inicializada) {
       
        printf("Arvore KD já inicializada. Reconstruindo...\n");
         _kdtree_destroi(arvore_global.raiz);
    }
    kdtree_constroi_arv(&arvore_global, comparador, distancia, EMBEDDING_DIM);
    arvore_inicializada = 1;
    printf("Arvore KD construida com k=%d\n", EMBEDDING_DIM);
}

tarv* get_tree() {
    if (!arvore_inicializada) {
       
        fprintf(stderr, "Error: Tentativa de obter arvore nao inicializada.\n");
     
        kdtree_construir();
    }
    return &arvore_global;
}

void inserir_ponto(treg ponto_reg) {
    if (!arvore_inicializada) {
        fprintf(stderr, "Atenção: Árvore não inicializada antes da inserção. Construindo agora...\n");
        kdtree_construir();
    }
    // Aloca memória para treg e copia os dados de ponto_reg
    treg *novo_ponto_heap = malloc(sizeof(treg));
    if (!novo_ponto_heap) {
        perror("Falha ao alocar memoria para novo ponto");
        return;
    }
    memcpy(novo_ponto_heap->embedding, ponto_reg.embedding, EMBEDDING_DIM * sizeof(float));
    strncpy(novo_ponto_heap->id, ponto_reg.id, ID_LEN -1);
    novo_ponto_heap->id[ID_LEN -1] = '\0';

    kdtree_insere(&arvore_global, novo_ponto_heap);
    printf("Ponto com ID '%s' inserido.\n", novo_ponto_heap->id);
}


treg buscar_mais_proximo(tarv *arv_ptr, treg query_reg) {
    if (!arv_ptr || !arv_ptr->raiz) {
        fprintf(stderr, "Erro: Árvore vazia ou não inicializada para busca.\n");
        treg not_found_reg;
        memset(&not_found_reg, 0, sizeof(treg));
        strncpy(not_found_reg.id, "NOT_FOUND", ID_LEN-1);
        not_found_reg.id[ID_LEN-1] = '\0';
        return not_found_reg;
    }

   
    tnode *mais_proximo_no = kdtree_busca_no(arv_ptr, &query_reg);

    if (mais_proximo_no != NULL && mais_proximo_no->key != NULL) {
        return *((treg *)(mais_proximo_no->key));
    } else {
        fprintf(stderr, "Nenhum ponto encontrado na busca.\n");
        treg not_found_reg;
        memset(&not_found_reg, 0, sizeof(treg));
        strncpy(not_found_reg.id, "NOT_FOUND", ID_LEN-1);
        not_found_reg.id[ID_LEN-1] = '\0';
        return not_found_reg;
    }
}
