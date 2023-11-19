@app.route('/analisis', methods=['POST'])
def insert_analisis_img():
    print(request.headers)
    print(request.content_type)
    #tipo="cerezas"
    
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)

    logger.debug("Headers de la solicitud: %s", request.headers)
    logger.debug("Tipo de contenido: %s", request.content_type)

    tipo = request.form.get('tipo')  # Usar request.form en lugar de request.json
    logger.debug("Tipo de fruta: %s", tipo)

    if tipo is None or len(tipo) == 0:
        logger.error("Error: tipo viene vacío o no se proporcionó")
        return "Error: tipo viene vacío o no se proporcionó"

    logger.debug("Tipo de fruta: %s", tipo)

    if 'imagen' not in request.files:
        logger.error("No se ha proporcionado un archivo.")
        return "No se ha proporcionado un archivo."


    #tipo = request.form.get('tipo')  # Usar request.form en lugar de request.json
    print(tipo)
    if tipo is None or len(tipo) == 0:
        print("Error: tipo viene vacío o no se proporcionó")
        return "Error: tipo viene vacío o no se proporcionó"
    
    print("Tipo de fruta:", tipo)
    
    if 'imagen' not in request.files:
        print("No se ha proporcionado un archivo.")
        return "No se ha proporcionado un archivo."
    
    # Obtener el archivo de la solicitud POST
    file = request.files['imagen']
    
    if file.filename == '':
        print("El archivo no tiene un nombre válido.")
        return "El archivo no tiene un nombre válido."
    
    if not file.content_type.startswith('image/'):
        print("El archivo no es una imagen válida.")
        return "El archivo no es una imagen válida."
    
    ruta_img = "image/"
    fecha_actual = datetime.datetime.now().strftime("%Y%m%d")
    numero_aleatorio = str(random.randint(1000, 9999))
    
    nombre_archivo = "img" + numero_aleatorio + "_" + fecha_actual + ".jpeg"
    
    file.save(ruta_img + nombre_archivo)
    
    print("Archivo guardado:", nombre_archivo)
    
    if tipo == "cerezas":
        proceso_analisis(ruta_img, nombre_archivo, numero_aleatorio, fecha_actual)
        print("Proceso de análisis de cerezas completado.")
        return "Se verificó el archivo y se generó un reporte con éxito para cerezas!"
    
    elif tipo == "arandanos":
        inicializar_arandanos(ruta_img, nombre_archivo, numero_aleatorio, fecha_actual)
        print("Proceso de análisis de arándanos completado.")
        return "Se verificó el archivo y se generó un reporte con éxito para arándanos!"
    else:
        print("El nombre no coincide ni con arándanos ni con cerezas.")
        return "El nombre no coincide ni con arándanos ni con cerezas."
 """













 """"
# Variables para almacenar los indicadores financieros INGRESOS Y GASTOS 
ingresos = 0
gastos = 0
activos = 0
pasivos = 0

# Función para registrar ingresos
def registrar_ingresos():
    global ingresos
    cantidad = float(input("Ingrese la cantidad de ingresos: $"))
    ingresos += cantidad

# Función para registrar gastos
def registrar_gastos():
    global gastos
    cantidad = float(input("Ingrese la cantidad de gastos: $"))
    gastos += cantidad

# Función para registrar activos
def registrar_activos():
    global activos
    cantidad = float(input("Ingrese la cantidad de activos: $"))
    activos += cantidad

# Función para registrar pasivos
def registrar_pasivos():
    global pasivos
    cantidad = float(input("Ingrese la cantidad de pasivos: $"))
    pasivos += cantidad

# Función para mostrar el resumen
def mostrar_resumen():
    print("\nResumen Financiero:")
    print()
    print(f"Ingresos totales: ${ingresos}")
    print()
    print(f"Gastos totales: ${gastos}")
    print()
    print(f"Ahorros-Activos  totales: ${activos}")
    print()
    print(f"Deudas-Pasivos totales: ${pasivos}")

# Menú principal
while True:
    print("\nSeleccione una opción:")
    print()
    print("1. Registrar Ingresos")
    print()
    print("2. Registrar Gastos")
    print()
    print("3. Registrar Ahorros - Activos ")
    print()
    print("4. Registrar Deudas - Pasivos")
    print()
    print("5. Mostrar Resumen")
    print()
    print("5. Mostrar Estado actual de la empresa")
    print()
    print("6. Salir")
    print()

    opcion = input("Opción: ")

    if opcion == "1":
        registrar_ingresos()
    elif opcion == "2":
        registrar_gastos()
    elif opcion == "3":
        registrar_activos()
    elif opcion == "4":
        registrar_pasivos()
    elif opcion == "5":
        mostrar_resumen()
    elif opcion == "6":
        break
    else:
        print("Opción no válida. Por favor, elija una opción válida.")

print("¡Gracias por usar el software FinanzasPro!")
""""