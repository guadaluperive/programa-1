import tkinter as tk
from tkinter import messagebox, ttk
from pulp import LpMaximize, LpProblem, LpStatus, LpVariable, lpSum, value
import matplotlib.pyplot as plt
import numpy as np

class SimplexGUI:
    def __init__(self, master):
        self.master = master
        self.master.title("Simplex Solver")
        
        # Variables de decisión y restricciones
        self.num_variables = 0
        self.num_constraints = 0
        
        # Frame para variables de decisión
        self.var_frame = ttk.LabelFrame(master, text="Variables de Decisión")
        self.var_frame.pack(padx=10, pady=10, fill=tk.BOTH)
        
        self.var_label = ttk.Label(self.var_frame, text="Número de Variables:")
        self.var_label.grid(row=0, column=0, padx=5, pady=5)
        
        self.var_entry = ttk.Entry(self.var_frame)
        self.var_entry.grid(row=0, column=1, padx=5, pady=5)
        
        # Frame para restricciones
        self.const_frame = ttk.LabelFrame(master, text="Restricciones")
        self.const_frame.pack(padx=10, pady=10, fill=tk.BOTH)
        
        self.const_label = ttk.Label(self.const_frame, text="Número de Restricciones:")
        self.const_label.grid(row=0, column=0, padx=5, pady=5)
        
        self.const_entry = ttk.Entry(self.const_frame)
        self.const_entry.grid(row=0, column=1, padx=5, pady=5)
        
        # Botón para confirmar variables y restricciones
        self.confirm_button = ttk.Button(master, text="Confirmar", command=self.confirm_variables_constraints)
        self.confirm_button.pack(padx=10, pady=10)
        
        # Solución
        self.obj_function = None
        self.constraints = []
        
    def confirm_variables_constraints(self):
        try:
            self.num_variables = int(self.var_entry.get())
            self.num_constraints = int(self.const_entry.get())
            
            self.create_input_widgets()
        except ValueError:
            messagebox.showerror("Error", "Ingrese números válidos para variables y restricciones.")
        
    def create_input_widgets(self):
        # Limpiar ventana actual
        for widget in self.master.winfo_children():
            widget.destroy()
        
        # Frame para la función objetivo
        self.obj_frame = ttk.LabelFrame(self.master, text="Función Objetivo")
        self.obj_frame.pack(padx=10, pady=10, fill=tk.BOTH)
        
        self.obj_label = ttk.Label(self.obj_frame, text="Función Objetivo (maximizar):")
        self.obj_label.grid(row=0, column=0, padx=5, pady=5)
        
        self.obj_vars = []
        self.obj_entries = []
        
        for i in range(self.num_variables):
            var_label = ttk.Label(self.obj_frame, text=f"x{i+1}:")
            var_label.grid(row=i, column=0, padx=5, pady=5, sticky=tk.W)
            
            var_entry = ttk.Entry(self.obj_frame)
            var_entry.grid(row=i, column=1, padx=5, pady=5)
            self.obj_entries.append(var_entry)
            self.obj_vars.append(LpVariable(f"x{i+1}", lowBound=None, cat='Continuous'))
        
        # Frame para las restricciones
        self.const_frame = ttk.LabelFrame(self.master, text="Restricciones")
        self.const_frame.pack(padx=10, pady=10, fill=tk.BOTH)
        
        self.constraints = []
        for i in range(self.num_constraints):
            const_label = ttk.Label(self.const_frame, text=f"Restricción {i+1}:")
            const_label.grid(row=i, column=0, padx=5, pady=5, sticky=tk.W)
            
            const_entry = ttk.Entry(self.const_frame, width=50)
            const_entry.grid(row=i, column=1, padx=5, pady=5)
            self.constraints.append(const_entry)
        
        # Botón para resolver
        solve_button = ttk.Button(self.master, text="Resolver", command=self.solve_problem)
        solve_button.pack(padx=10, pady=10)
        
    def solve_problem(self):
        problem = LpProblem("SimplexProblem", LpMaximize)
        
        # Función objetivo
        try:
            obj_expr = lpSum(self.obj_vars[i] * float(entry.get()) for i, entry in enumerate(self.obj_entries))
            problem += obj_expr
        except Exception as e:
            messagebox.showerror("Error", f"Error en la función objetivo: {str(e)}")
            return
        
        # Restricciones
        for j, entry in enumerate(self.constraints):
            try:
                # Parsear la restricción ingresada por el usuario
                constraint_text = entry.get().strip()
                
                # Validar que la restricción no esté vacía
                if not constraint_text:
                    raise ValueError("Ingrese una restricción válida.")
                
                # Separar la restricción en coeficientes y variables
                constraint_parts = constraint_text.split()
                coeficients = []
                variables = []
                sign = 1  # To handle signs in the constraint
                
                for part in constraint_parts:
                    if part.replace(".", "", 1).isdigit():  # Check if the part is a number
                        coeficients.append(float(part) * sign)
                        sign = 1  # Reset sign after the number
                    elif part == "-":
                        sign = -1  # Set sign to negative after "-"
                    elif part.startswith("x"):
                        variables.append(part)
                
                # Build the constraint expression
                constraint_expr = lpSum(coeficients[i] * self.obj_vars[int(variables[i][1:]) - 1] for i in range(len(variables)))
                
                # Add the constraint to the problem
                problem += constraint_expr <= float(coeficients[-1])
            
            except Exception as e:
                messagebox.showerror("Error", f"Error en la restricción {j+1}: {str(e)}")
                return
        
        # Resolve the problem
        problem.solve()
        
        # Show results
        self.show_results(problem)

    def show_results(self, problem):
        # Crear ventana para resultados
        result_window = tk.Toplevel(self.master)
        result_window.title("Resultado")
        
        status = LpStatus[problem.status]
        if status == "Optimal":
            # Mostrar valor óptimo encontrado
            result_label = ttk.Label(result_window, text=f"Valor óptimo encontrado: {value(problem.objective)}")
            result_label.pack(padx=10, pady=10)
            
            # Mostrar solución
            solution_label = ttk.Label(result_window, text="Solución:")
            solution_label.pack(padx=10, pady=5)
            
            for var in problem.variables():
                var_label = ttk.Label(result_window, text=f"{var.name} = {var.varValue}")
                var_label.pack(padx=10, pady=2, anchor=tk.W)
            
            # Mostrar gráfica (solo si hay 2 variables de decisión)
            if len(problem.variables()) == 2:
                self.plot_graph(problem, result_window)
        else:
            result_label = ttk.Label(result_window, text=f"Estado de la solución: {status}")
            result_label.pack(padx=10, pady=10)
    
    def plot_graph(self, problem, result_window):
        # Extraer coeficientes de la función objetivo
        coef_x1 = problem.variables()[0].varValue
        coef_x2 = problem.variables()[1].varValue
        
        # Definir puntos para graficar la función objetivo
        x1_values = np.linspace(0, 10, 100)
        x2_values = np.linspace(0, 10, 100)
        Z_values = np.zeros((100, 100))
        
        for i, x1 in enumerate(x1_values):
            for j, x2 in enumerate(x2_values):
                Z_values[i, j] = coef_x1 * x1 + coef_x2 * x2
        
        # Graficar la función objetivo
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection='3d')
        X, Y = np.meshgrid(x1_values, x2_values)
        ax.plot_surface(X, Y, Z_values, cmap='viridis', alpha=0.8)
        
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        ax.set_zlabel('Z')
        ax.set_title('Función Objetivo Maximizada')
        
        # Mostrar la gráfica en la ventana de resultados
        canvas = plt.backends.backend_tkagg.FigureCanvasTkAgg(fig, master=result_window)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

def main():
    root = tk.Tk()
    simplex_gui = SimplexGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
