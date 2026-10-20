import numpy as np
import re

class FuzzySet:
    def __init__(self, name, f_type, params):
        self.name = name
        self.f_type = f_type
        self.params = [float(p) for p in params]

    def membership(self, x):
        a, b, c = self.params
        if self.f_type == "Треугольник":
            if x <= a or x >= c:
                return 0.0
            elif a < x <= b:
                return (x - a) / (b - a) if b != a else 1.0
            else: # b < x < c
                return (c - x) / (c - b) if c != b else 1.0
        return 0.0

class FuzzyVariable:
    def __init__(self, name, min_val, max_val):
        self.name = name
        self.min_val = min_val
        self.max_val = max_val
        self.terms = {}

    def fuzzify(self, value):
        result = {}
        for name, fs in self.terms.items():
            result[name] = fs.membership(value)
        return result

class Rule:
    def __init__(self, conditions, conclusion):
        self.conditions = conditions 
        self.conclusion = conclusion 

    def get_weight(self, fuzzified_inputs, agg_op='min'):
        weights = []
        for var, term in self.conditions:
            w = fuzzified_inputs.get(var, {}).get(term, 0.0)
            weights.append(w)
        if not weights:
            return 0.0
        if agg_op == 'min':
            return min(weights)
        elif agg_op == 'prod':
            return np.prod(weights)
        return 0.0

class FuzzySystem:
    def __init__(self):
        self.variables = {}
        self.rules = []

    def load_from_file(self, filename):
        current_var = None
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split()
                if parts[0] == 'VAR':
                    var_name = parts[1]
                    min_val, max_val = float(parts[2]), float(parts[3])
                    current_var = FuzzyVariable(var_name, min_val, max_val)
                    self.variables[var_name] = current_var
                elif parts[0] == 'TERM':
                    term_name, f_type = parts[1], parts[2]
                    params = parts[3:]
                    if current_var:
                        current_var.terms[term_name] = FuzzySet(term_name, f_type, params)
                elif parts[0] == 'ЕСЛИ':
                    line = line.replace('ЕСЛИ ', '').replace(' ТО ', ' ТО ')
                    if ' ТО ' in line:
                        cond_str, concl_str = line.split(' ТО ')
                        conditions = []
                        for c in cond_str.split(' И '):
                            var, term = c.split('=')
                            conditions.append((var.strip(), term.strip()))
                        var, term = concl_str.split('=')
                        conclusion = (var.strip(), term.strip())
                        self.rules.append(Rule(conditions, conclusion))

    def infer(self, inputs, implication='min', aggregation='max', defuzz='centroid', mechanism='mamdani'):
        fuzzified_inputs = {}
        for var_name, value in inputs.items():
            if var_name in self.variables:
                fuzzified_inputs[var_name] = self.variables[var_name].fuzzify(value)
        out_var_name = self.rules[0].conclusion[0]
        out_var = self.variables[out_var_name]
        if mechanism == 'larsen':
            num = 0.0
            den = 0.0
            for rule in self.rules:
                w = rule.get_weight(fuzzified_inputs, agg_op='min')
                if w > 0:
                    term_name = rule.conclusion[1]
                    fs = out_var.terms[term_name]
                    center = fs.params[1] 
                    num += w * center
                    den += w
            return num / den if den > 0 else 0.0
        else:
            x_vals = np.linspace(out_var.min_val, out_var.max_val, 500)
            aggregated = np.zeros_like(x_vals)
            for rule in self.rules:
                w = rule.get_weight(fuzzified_inputs, agg_op='min')
                if w > 0:
                    term_name = rule.conclusion[1]
                    fs = out_var.terms[term_name]
                    y_vals = np.array([fs.membership(x) for x in x_vals])
                    if implication == 'min':
                        y_vals = np.minimum(y_vals, w)
                    elif implication == 'prod':
                        y_vals = y_vals * w
                    if aggregation == 'max':
                        aggregated = np.maximum(aggregated, y_vals)
                    elif aggregation == 'sum':
                        aggregated = aggregated + y_vals
            if defuzz == 'centroid':
                if np.sum(aggregated) == 0:
                    return 0.0
                return np.sum(x_vals * aggregated) / np.sum(aggregated)
            return 0.0

def main():
    fs = FuzzySystem()
    fs.load_from_file('rules.txt')
    print(f"Загружено правил: {len(fs.rules)}")
    print()
    while True:
        print("\n--- МЕНЮ ---")
        print("1. Тест 1: 1 вход (Температура)")
        print("2. Тест 2: 2 входа (Температура + Влажность)")
        print("3. Тест 3: 3 входа (Температура + Влажность + Загрузка)")
        print("4. Ввести свои значения вручную")
        print("5. Выход")     
        choice = input("Ваш выбор (1-5): ").strip()
        if choice == "1":
            t = float(input("Введите температуру (-20...50): "))
            inputs = {'Температура': t}
            run_all_mechanisms(fs, inputs)
        elif choice == "2":
            t = float(input("Введите температуру (-20...50): "))
            v = float(input("Введите влажность (0...100): "))
            inputs = {'Температура': t, 'Влажность': v}
            run_all_mechanisms(fs, inputs)
        elif choice == "3":
            t = float(input("Введите температуру (-20...50): "))
            v = float(input("Введите влажность (0...100): "))
            z = float(input("Введите загрузку двигателя (0...100): "))
            inputs = {'Температура': t, 'Влажность': v, 'Загрузка_двигателя': z}
            run_all_mechanisms(fs, inputs)
        elif choice == "4":
            inputs = {}
            print("\nДоступные переменные:")
            for var in fs.variables:
                if var != 'Мощность_кондиционера': # выход не вводим
                    print(f" - {var} ({fs.variables[var].min_val}...{fs.variables[var].max_val})")
            t = input("Температура (Enter - пропустить): ").strip()
            if t: inputs['Температура'] = float(t)
            v = input("Влажность (Enter - пропустить): ").strip()
            if v: inputs['Влажность'] = float(v)
            z = input("Загрузка двигателя (Enter - пропустить): ").strip()
            if z: inputs['Загрузка_двигателя'] = float(z)
            if inputs:
                run_all_mechanisms(fs, inputs)
            else:
                print("Нет введенных данных.")
        elif choice == "5":
            print("Выход.")
            break
        else:
            print("Неверный выбор.")
def run_all_mechanisms(fs, inputs):
    """Вспомогательная функция для вывода результатов всех 4 механизмов"""
    print(f"\nВходные данные: {inputs}")
    print("-" * 55)
    res_min = fs.infer(inputs, implication='min', aggregation='max')
    res_prod = fs.infer(inputs, implication='prod', aggregation='max')
    res_sum = fs.infer(inputs, implication='min', aggregation='sum')
    res_larsen = fs.infer(inputs, mechanism='larsen')
    print(f"1. Импликация MIN, Агрегация MAX (Мамдани): {res_min:.2f}")
    print(f"2. Импликация PROD, Агрегация MAX (Мамдани): {res_prod:.2f}")
    print(f"3. Импликация MIN, Агрегация SUM (Альтернатива): {res_sum:.2f}")
    print(f"4. Уровни истинности (Ларсен): {res_larsen:.2f}")
    print("-" * 55)
if __name__ == "__main__":
    main()