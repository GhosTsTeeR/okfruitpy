from prettytable import PrettyTable
import matplotlib.pyplot as plt
import numpy as np

# Variables para almacenar los indicadores financieros
ingresos_mensuales = [0] * 12
gastos_mensuales = [0] * 12
activos_mensuales = [0] * 12
pasivos_mensuales = [0] * 12

# Función para registrar los indicadores financieros
def registrar_datos_mensuales():
    mes = int(input("Ingrese el número de mes (1-12): "))
    mes -= 1  # Restamos 1 para ajustar al índice de la lista
    ingresos_mensuales[mes] += float(input(f"Ingrese la cantidad de ingresos para el mes {mes + 1}: $"))
    gastos_mensuales[mes] += float(input(f"Ingrese la cantidad de gastos para el mes {mes + 1}: $"))
    activos_mensuales[mes] += float(input(f"Ingrese la cantidad de activos para el mes {mes + 1}: $"))
    pasivos_mensuales[mes] += float(input(f"Ingrese la cantidad de pasivos para el mes {mes + 1}: $"))

# Función para mostrar el resumen de un mes específico
def mostrar_resumen_mensual(mes):
    resumen = PrettyTable()
    resumen.field_names = ["Concepto", "Cantidad"]
    resumen.add_row(["Ingresos", f"${ingresos_mensuales[mes]:.2f}"])
    resumen.add_row(["Gastos", f"${gastos_mensuales[mes]:.2f}"])
    resumen.add_row(["Ahorros-Activos", f"${activos_mensuales[mes]:.2f}"])
    resumen.add_row(["Deudas-Pasivos", f"${pasivos_mensuales[mes]:.2f}"])
    resumen.add_row(["Saldo Final", f"${ingresos_mensuales[mes] - gastos_mensuales[mes] - pasivos_mensuales[mes] + activos_mensuales[mes]:.2f}"])
    
    print(f"\nResumen Financiero del Mes ({mes + 1}):")
    print(resumen)

# Función para mostrar el resumen acumulado de todos los meses
def mostrar_resumen_acumulado():
    resumen_acumulado = PrettyTable()
    resumen_acumulado.field_names = ["Concepto", "Cantidad"]
    resumen_acumulado.add_row(["Ingresos totales", f"${sum(ingresos_mensuales):.2f}"])
    resumen_acumulado.add_row(["Gastos totales", f"${sum(gastos_mensuales):.2f}"])
    resumen_acumulado.add_row(["Ahorros-Activos totales", f"${sum(activos_mensuales):.2f}"])
    resumen_acumulado.add_row(["Deudas-Pasivos totales", f"${sum(pasivos_mensuales):.2f}"])
    resumen_acumulado.add_row(["Saldo Final acumulado", f"${sum(ingresos_mensuales) - sum(gastos_mensuales) - sum(pasivos_mensuales) + sum(activos_mensuales):.2f}"])

    print("\nResumen Financiero Acumulado:")
    print(resumen_acumulado)

# Función para mostrar gráficamente el resumen de un mes específico
def mostrar_grafica_resumen_mensual(mes):
    conceptos = ["Ingresos", "Gastos", "Ahorros-Activos", "Deudas-Pasivos", "Saldo Final"]
    cantidades = [ingresos_mensuales[mes], gastos_mensuales[mes], activos_mensuales[mes], pasivos_mensuales[mes],
                  ingresos_mensuales[mes] - gastos_mensuales[mes] - pasivos_mensuales[mes] + activos_mensuales[mes]]

    plt.bar(conceptos, cantidades, color=['green', 'red', 'blue', 'orange', 'purple'])
    plt.title(f"Resumen Financiero del Mes ({mes + 1})")
    plt.xlabel("Concepto")
    plt.ylabel("Cantidad ($)")
    plt.show()

# Menú principal
while True:
    print("\nSeleccione una opción:")
    print("1. Registrar Datos Mensuales")
    print("2. Mostrar Resumen Mensual")
    print("3. Mostrar Resumen Acumulado")
    print("4. Ver Resultados Mensuales en Gráfica")
    print("5. Salir")

    opcion = input("Opción: ")

    if opcion == "1":
        registrar_datos_mensuales()
    elif opcion == "2":
        mes_mostrar = int(input("Ingrese el número de mes para mostrar el resumen: "))
        mostrar_resumen_mensual(mes_mostrar - 1)
    elif opcion == "3":
        mostrar_resumen_acumulado()
    elif opcion == "4":
        mes_grafica = int(input("Ingrese el número de mes para mostrar la gráfica: "))
        mostrar_grafica_resumen_mensual(mes_grafica - 1)
    elif opcion == "5":
        break
    else:
        print("Opción no válida. Por favor, elija una opción válida.")

print("¡Gracias por usar el software de indicadores financieros!")
