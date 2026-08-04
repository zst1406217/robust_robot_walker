<!-- <a href=https://arxiv.org/abs/2406.09402><img src='https://img.shields.io/badge/arXiv-2402.00752-b31b1b.svg'></a> <a href='https://immortalco.github.io/Instruct-4D-to-4D/'><img src='https://img.shields.io/badge/Project-Page-Green'></a>  -->

<div align="center">

# Robust Robot Walker: Aprendiendo Locografía Ágil sobre Terrenos Desafiantes

[![Paper](https://img.shields.io/badge/arXiv-2409.07409-brightgreen)](https://arxiv.org/abs/2409.07409) [![Conference](https://img.shields.io/badge/ICRA-2025-blue)](https://2025.ieee-icra.org/) [![Project WebPage](https://img.shields.io/badge/Project-webpage-%23fc4d5d)](https://robust-robot-walker.github.io/) [![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

</div>

![Pipeline](./asset/pipeline-1.png)
> [Robust Robot Walker: Aprendiendo Locografía Ágil sobre Terrenos Desafiantes](https://arxiv.org/abs/2409.07409) \
> Shaoting Zhu, Runhan Huang, Linzhan Mou, Hang Zhao \
> ICRA 2025

## Instalación

### 1. Crear un Entorno Virtual de Python
- Cree un nuevo entorno virtual de Python con Python 3.6, 3.7 o 3.8 (se recomienda Python 3.8).

### 2. Instalar PyTorch 1.10 con CUDA 11.3
Ejecute el siguiente comando para instalar PyTorch:

```bash
pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
```

### 3. Instalar Isaac Gym
- Descargue e instale **Isaac Gym Preview 4** desde [NVIDIA Isaac Gym](https://developer.nvidia.com/isaac-gym).
- Instale las dependencias necesarias de Python:

```bash
cd isaacgym/python && pip install -e .
```

- Pruébelo ejecutando un ejemplo:

```bash
cd examples && python 1080_balls_of_solitude.py
```

- Para solucionar problemas, consulte la documentación de Isaac Gym en `isaacgym/docs/index.html`.

### 4. Clonar Este Repositorio
Clone el repositorio con el siguiente comando:

```bash
git clone https://github.com/zst1406217/robust_robot_walker.git
cd robust_robot_walker
```

### 5. Instalar `rsl_rl` (Implementación PPO)
```bash
cd rsl_rl && pip install -e .
```

### 6. Instalar `legged_gym`
```bash
cd legged_gym && pip install -e .
```

### 7. Instalar Dependencias Adicionales
Instale las dependencias necesarias:

```bash
pip install debugpy tqdm numpy==1.19.5 tensorboard==2.0.0 setuptools==58.5.0 protobuf==3.20.0 matplotlib==3.4.0
```

### 8. Descargar el Conjunto de Datos AMP
Descargue y coloque el archivo [mpc_data_no_command.npy](https://drive.google.com/file/d/1zWMzw6EK7PKM8V5u1HQ4SrbvkIZ-veHG/view?usp=sharing) en `./legged_gym`.

## Ejecutar el Benchmark Tiny-Trap
1. Navegue a la carpeta `legged_gym`:

```bash
cd legged_gym
```

2. La política entrenada está almacenada en `checkpoint.zip`. Descomprímala y coloque la carpeta en `legged_gym/logs/rrw_a1`.

3. Asegúrese de estar en el directorio `robust_robot_walker/legged_gym` y ejecute el benchmark:

```bash
python legged_gym/scripts/track.py --task a1_bartrack --load_run checkpoint --headless
```

Esto mostrará la tasa de éxito, el tiempo promedio de paso y la distancia promedio recorrida de 1,000 robots. Debido a la semilla aleatoria y las variaciones en los dispositivos GPU, los resultados pueden diferir ligeramente del artículo.

## Entrenamiento de la Política


### 1. Entrenar la Política de Caminata en un Planal
Para entrenar la política de caminata en un plano plano:

```bash
cd legged_gym
```

Asegúrese de estar en el directorio `robust_robot_walker/legged_gym` y luego ejecute:

```bash
python legged_gym/scripts/train.py --task a1_remotegoal --headless
```

### 2. Entrenar la Política Robust-Robot-Walker (Etapa 1)
1. Actualice el parámetro `"load_run"` en el archivo `legged_gym/envs/a1/a1_mix_goal_stage1_config.py` (línea 182) con el directorio de registro de la Etapa 1 (Entrenar la política de caminata en el plano).

   Por ejemplo:

   ```python
   load_run = "Mar14_12-59-47_WalkByRemoteGoal_noResume"
   ```

2. Asegúrese de estar en la carpeta `robust_robot_walker/legged_gym` y ejecute:

```bash
python legged_gym/scripts/train.py --task a1_mixgoalstage1 --headless
```

### 3. Entrenar la Política Robust-Robot-Walker (Etapa 2)
1. Actualice el parámetro `"load_run"` en el archivo `legged_gym/envs/a1/a1_mix_goal_stage2_config.py` (línea 182) con el directorio de registro de la Etapa 2 (Entrenar la política Robust-Robot-Walker, etapa 1).

2. Ejecute lo siguiente:

```bash
python legged_gym/scripts/train.py --task a1_mixgoalstage2 --headless
```

## Prueba y Visualización de la Política de Caminata en un Planos
Para probar y visualizar la política de caminata:

```bash
cd legged_gym
python legged_gym/scripts/play.py --task a1_mixgoalstage2 --load_run <su_directorio_de_registro>
```

## Implementación en un Robot Real
Consulte [Deploy.md](./Deploy.md) para instrucciones sobre cómo implementarlo en un robot real.

## Agradecimiento
https://github.com/leggedrobotics/legged_gym<br>
https://github.com/ZiwenZhuang/parkour

## Cita

Puede encontrar nuestro artículo en [arXiv](https://arxiv.org/abs/2409.07409).

Si encuentra que este código o el artículo es útil para su investigación, por favor considere citar:

```text
@article{zhu2024robust,
  title={Robust Robot Walker: Learning Agile Locomotion over Tiny Traps},
  author={Shaoting, Zhu and Runhan, Huang and Linzhan, Mou and Hang, Zhao},
  journal={arXiv preprint arXiv:2409.07409},
  year={2024}
}
```

---
<!-- Traducción completada al español, preservando comandos, enlaces, identificadores y estructura Markdown. -->
