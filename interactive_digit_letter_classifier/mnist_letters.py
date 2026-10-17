import pandas as pd
from models_pya.ml import MLP, Regresion_Softmax
import sys
import numpy as np
import pygame

data = pd.read_csv("data/emnist_letters_test.csv")

datos = data.iloc[:]
datos = datos.to_numpy(dtype=int)

# 1. Inicialización de Pygame
pygame.init()
pygame.font.init()

# 2. Configuración de constantes generales
GRID_SIZE = 28  # Matriz de 28x28 celdas
CELL_SIZE = 18  # Tamaño de celda en píxeles para el lienzo
CANVAS_SIZE = GRID_SIZE * CELL_SIZE  # 504x504 px

WINDOW_WIDTH = 1250
WINDOW_HEIGHT = 650

# Offsets del lienzo principal
OFFSET_X = 50
OFFSET_Y = 100

# Colores (R, G, B)
COLOR_BG = (35, 35, 35)
COLOR_CANVAS_BG = (15, 15, 15)
COLOR_GRID = (60, 60, 60)
COLOR_TEXT = (230, 230, 230)
COLOR_TEXT_PROB = (220, 220, 220)
COLOR_BAR_BG = (40, 40, 45)
COLOR_BAR_FILL = (0, 180, 216)  # Color cian para probabilidades

# Parametros
params = np.load(r'interactive_digit_letter_classifier\params_letters.npz')

# Modelos
softmax = Regresion_Softmax(50, 0.1, architecture=(784,26), mini_batch=(True, 128))
softmax._pesos = params['soft_w']
softmax._sesgos = params['soft_b']

architecture_nn = params['mlp_arch']
pesos_nn = [params[f'mlp_w{peso}'] for peso in range(1, len(architecture_nn) )]
sesgos_nn = [params[f'mlp_b{sesgo}'] for sesgo in range(1, len(architecture_nn))]


nn = MLP(10, 0.1, architecture=architecture_nn, loss=['log'], activation=params['mlp_act'], optimizer='batch')
nn._pesos = pesos_nn
nn._sesgos = sesgos_nn


# Modelos a mostrar
modelos = ["RED NEURONAL", "SOFTMAX"]
precision = {modelo: 0.0 for modelo in modelos}

X = datos[:, 1:] / 255
y = datos[:, 0]


preds_soft = np.argmax(softmax.f_hat(X), axis=1)
res_soft = (y == preds_soft).astype(dtype=int)

preds_nn = np.argmax(nn.f_hat(X), axis=1)
res_nn = (y == preds_nn).astype(dtype=int)

precision['SOFTMAX'] = round(np.sum(res_soft) / len(res_soft) * 100, 2)
precision['RED NEURONAL'] = round(np.sum(res_nn) / len(res_nn) * 100, 2)

# Letras posibles
letters_pred = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']

# Fuentes
font_title = pygame.font.SysFont("arial", 26, bold=True)
font_numbers = pygame.font.SysFont("arial", 25, bold=True)
font_models = pygame.font.SysFont("arial", 30, bold=True)
font_probability = pygame.font.SysFont('arial', 18, bold=True)

# Estructura para almacenar probabilidades simuladas para cada modelo (10 dígitos cada uno)
# En una aplicación real, aquí insertarás las predicciones de tu modelo de IA.
probabilidades = {modelo: np.zeros(26) for modelo in modelos}

# --- Configuración del Panel de Predicciones ---
START_PRED_X = OFFSET_X + CANVAS_SIZE + 100  # Iniciar a la derecha del lienzo
START_PRED_Y = OFFSET_Y

DIGIT_BOX_SIZE = 45  # Tamaño de cada casilla (ancho y alto) de los dígitos 0-9
BAR_WIDTH = 100  # Ancho de la barra de probabilidad
COLUMN_GAP = 330  # Espacio horizontal entre cada bloque de modelo

screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Paint 28x28 - Predicciones")

# Matriz del lienzo de dibujo
grid = np.array([[0.0 for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)])
flat = grid.reshape(-1)

probabilidades['REGRESION SOFTMAX'] = softmax.f_hat(x=flat) * 100
probabilidades['RED NEURONAL'] = nn.f_hat(x=flat) * 100




def apply_brush(mouse_x, mouse_y):
    local_x = mouse_x - OFFSET_X
    local_y = mouse_y - OFFSET_Y

    if 0 <= local_x < CANVAS_SIZE and 0 <= local_y < CANVAS_SIZE:
        center_col = local_x // CELL_SIZE
        center_row = local_y // CELL_SIZE
        radius = 1.6
        round_radius = round(radius)

        for r in range(
            max(0, center_row - round_radius), min(GRID_SIZE, center_row + round_radius + 1)
        ):
            for c in range(
                max(0, center_col - round_radius),
                min(GRID_SIZE, center_col + round_radius + 1),
            ):
                distance = np.sqrt(
                    (r - center_row) ** 2 + (c - center_col) ** 2
                )
                if distance <= radius:
                    intensity = 1.0 - (distance / radius)
                    grid[r][c] = min(1.0, grid[r][c] + intensity * 0.3)


def delete_brush(mouse_x, mouse_y):
    local_x = mouse_x - OFFSET_X
    local_y = mouse_y - OFFSET_Y

    if 0 <= local_x < CANVAS_SIZE and 0 <= local_y < CANVAS_SIZE:
        center_col = local_x // CELL_SIZE
        center_row = local_y // CELL_SIZE
        radius = 1.25
        round_radius = round(radius)

        for r in range(
            max(0, center_row - round_radius), min(GRID_SIZE, center_row + round_radius + 1)
        ):
            for c in range(
                max(0, center_col - round_radius),
                min(GRID_SIZE, center_col + round_radius + 1),
            ):
                distance = np.sqrt(
                    (r - center_row) ** 2 + (c - center_col) ** 2
                )
                if distance <= radius:
                    intensity = 1.0 - (distance / radius)
                    grid[r][c] = max(0.0, grid[r][c] - intensity * 0.3)


# Bucle Principal
running = True
drawing = False
borranding = False


while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                drawing = True
                apply_brush(*event.pos)
            elif event.button == 3:
                borranding = True
                delete_brush(*event.pos)

        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:
                drawing = False
            elif event.button == 3:
                borranding = False

        elif event.type == pygame.MOUSEMOTION:
            if drawing:
                apply_brush(*event.pos)
            elif borranding:
                delete_brush(*event.pos)

        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_c:
                grid = np.asarray([
                    [0.0 for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)
                ])
                flat = grid.reshape(-1)
            elif event.key == pygame.K_RETURN:
                idx = np.random.randint(0, len(datos))
                grid = datos[idx][1:].reshape((28,28)) / 255
                flat = grid.reshape(-1)

    screen.fill(COLOR_BG)

    # 1. Título General
    title_surface = font_title.render("Predicciones", True, COLOR_TEXT)
    screen.blit(title_surface, (START_PRED_X, 25))

    # 2. Dibujar Lienzo Principal
    pygame.draw.rect(
        screen,
        COLOR_CANVAS_BG,
        (OFFSET_X, OFFSET_Y, CANVAS_SIZE, CANVAS_SIZE),
    )
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            val = grid[r][c]
            x = c * CELL_SIZE + OFFSET_X
            y = r * CELL_SIZE + OFFSET_Y

            if val > 0:
                color_val = int(val*255)

                pygame.draw.rect(
                    screen,
                    (color_val, color_val, color_val),
                    (x, y, CELL_SIZE, CELL_SIZE),
                )
            pygame.draw.rect(
                screen, COLOR_GRID, (x, y, CELL_SIZE, CELL_SIZE), 1
            )


    # probabilidades
    probabilidades['SOFTMAX'] = softmax.f_hat(x=flat) * 100
    probabilidades['RED NEURONAL'] = nn.f_hat(x=flat) * 100

    high_prob = [np.argmax(probabilidades[model]) for model in modelos]

    

    # 3. Dibujar Secciones de Modelos con Cuadrículas y Probabilidades
    for idx_model, model_name in enumerate(modelos):
        # Calcular coordenada X base para este modelo
        col_x = START_PRED_X + (idx_model * COLUMN_GAP)

        idx_probs = np.argpartition(probabilidades[model_name], -10)[-10:]

        # --- A. Cuadrícula de Números (0-9) y Barras de Probabilidad ---
        for letter_idx, digit in zip(idx_probs, np.arange(10)):
            box_y = START_PRED_Y + (digit * (DIGIT_BOX_SIZE + 5))

            # Cuadrícula para el número
            num_rect = pygame.Rect(
                col_x, box_y, DIGIT_BOX_SIZE, DIGIT_BOX_SIZE
            )
            pygame.draw.rect(screen, COLOR_CANVAS_BG, num_rect)
            pygame.draw.rect(screen, COLOR_GRID, num_rect, 1)

            # Texto del número
            num_surface = font_numbers.render(letters_pred[letter_idx], True, COLOR_TEXT)
            screen.blit(num_surface, num_surface.get_rect(center=num_rect.center))

            # Barra de probabilidad (Cuadrícula contigua a la derecha)
            bar_x = col_x + DIGIT_BOX_SIZE + 20
            bar_rect = pygame.Rect(bar_x, box_y + 8, BAR_WIDTH, 35)
            pygame.draw.rect(screen, COLOR_BAR_BG, bar_rect)
            pygame.draw.rect(screen, COLOR_GRID, bar_rect, 1)

            # Relleno según el nivel de probabilidad (ej. 0.0 a 1.0)
            prob_val = probabilidades[model_name][letter_idx]
            if prob_val > 0:
                fill_width = int(BAR_WIDTH * prob_val / 100)
                fill_rect = pygame.Rect(bar_x, box_y + 8, fill_width, 35)

                if high_prob[idx_model] == letter_idx: 
                    pygame.draw.rect(screen, (220, 100, 120), fill_rect)
                else:
                    pygame.draw.rect(screen, COLOR_BAR_FILL, fill_rect)


            prob_surface = font_probability.render(f"{prob_val:.2f}%", True, COLOR_TEXT_PROB)
            prob_rect = prob_surface.get_rect(center=(bar_x + 35, box_y + 25))
            screen.blit(prob_surface, prob_rect)


        # --- B. Nombre del Modelo Vertical ---
        # Posicionado entre la casilla del número y la barra de probabilidad
        model_text_x = col_x - 25
        letters = list(model_name)
        total_text_height = len(letters) * 14
        start_text_y = (
            START_PRED_Y + (7 * (DIGIT_BOX_SIZE)) // 2
        ) - (total_text_height // 2)

        for j, letter in enumerate(letters):
            char_surface = font_models.render(letter, True, COLOR_TEXT)
            char_rect = char_surface.get_rect(
                center=(model_text_x, start_text_y + (j * 30))
            )
            screen.blit(char_surface, char_rect)

    for i,m in enumerate(modelos):
        prec_surface = font_probability.render(f'Precision: {precision[m]}%', True, COLOR_TEXT)
        prec_rect = prec_surface.get_rect(center=(680 + (i * COLUMN_GAP), 620))

        screen.blit(prec_surface, prec_rect)

    pygame.display.flip()

pygame.quit()
sys.exit()