import ctypes
from ctypes import Structure, POINTER, c_float, c_int, c_char, c_double

# Constants from C (important to keep in sync)
EMBEDDING_DIM = 128
ID_LEN = 100

class TReg(Structure):
    _fields_ = [("embedding", c_float * EMBEDDING_DIM),
                ("id", c_char * ID_LEN)
               ]

class TNode(Structure):
    pass

TNode._fields_ = [("key", ctypes.c_void_p),
                  ("esq", POINTER(TNode)),
                  ("dir", POINTER(TNode))]

class Tarv(Structure):
    _fields_ = [("raiz", POINTER(TNode)),
                
                ("cmp", ctypes.c_void_p),
                ("dist", ctypes.c_void_p), 
                ("k", c_int)
               ]

# Carregar a biblioteca C

try:
    lib = ctypes.CDLL("./libkdtree.so")
except OSError as e:
    print(f"Error loading libkdtree.so: {e}")
    print("Make sure you have compiled kdtree.c (e.g., using: gcc -shared -o libkdtree.so -fPIC kdtree.c)")
    exit(1)


# Definir a assinatura das funções da API C
lib.kdtree_construir.argtypes = []
lib.kdtree_construir.restype = None

lib.get_tree.argtypes = []
lib.get_tree.restype = POINTER(Tarv)

lib.inserir_ponto.argtypes = [TReg] 
lib.inserir_ponto.restype = None

lib.buscar_mais_proximo.argtypes = [POINTER(Tarv), TReg] 
lib.buscar_mais_proximo.restype = TReg
