from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import itertools
import random

from modules.utils import load_latest_dataset_from_storage, load_best_model_from_storage

class OrderPrediction_Frontend():
    
    @property
    def app(self) -> Flask:
        return self.__app
    
    @app.setter
    def app(self, app: Flask):
        self.__app = app
        
    def __init__(self, model_directory, data_directory, data_preproc_directory ,feature_directory) -> None:
        self.__app = Flask(__name__)
        self.model_directory = model_directory
        self.data_directory = data_directory
        self.data_preproc_directory = data_preproc_directory
        self.feature_directory = feature_directory
        
        # Load the models and files
        self.models = {target: load_best_model_from_storage(self.model_directory, target) for target in ['OEE', 'AVAIL', 'PERF', 'QUAL', 'APT', 'PBT']}
        self.latest_product_ref = load_latest_dataset_from_storage(self.data_directory, 'Product_Reference').sort_values(by=['ProductCode'])
        self.latest_product_enc = load_latest_dataset_from_storage(self.data_directory, 'ProductEncoding_Reference').sort_values(by=['ProductCode'])
        self.latest_line_enc = load_latest_dataset_from_storage(self.data_directory, 'LineEncoding_Reference').sort_values(by=['Code'])
        self.latest_change_feature = load_latest_dataset_from_storage(self.feature_directory, 'ProductChange_Category_Features')
        self.latest_historic_times_feature = load_latest_dataset_from_storage(self.feature_directory, 'Product_HistoricTimes_Features')
        self.latest_encoded_product_features = load_latest_dataset_from_storage(self.data_preproc_directory, 'Encoded_Feature_Dataset')
        self.product_codes_list = self.latest_product_enc['ProductCode'].dropna().unique().tolist()
        self.line_codes_list = self.latest_line_enc['Code'].dropna().unique().tolist()

        ## Init Variables
        self.predictions = {}
        self.prediction_made = False  # Flag to check if a prediction was made
        self.alert = None
        self.compare_results = None
        
    def register_endpoints(self):
        self.__app.add_url_rule(rule='/', endpoint='index', view_func=self.index, methods=['GET', 'POST'])
        
    def run(self, *args, **kwargs):
        self.__app.run(*args, **kwargs)

    # Rendering Method
    def index(self):
        predictions = {}
        prediction_made = False
        alert = None
        compare_results = None
        best_order = None
        original_order = None
        original_summary = None
        optimal_summary = None
        best_order_dict = None

        if request.method == 'POST':
            
            form_type = request.form['form_type']
            
            if form_type == 'single_prediction':
                # Single Prediction Logic
                artikel = request.form['artikel']
                maschine = request.form['maschine']
                auftragsgroesse = request.form['auftragsgroesse']
                vorgaenger_artikel = request.form['vorgaenger_artikel']
                produkt_wechsel = vorgaenger_artikel + '-' + artikel

                input_data = self.get_input_sample(artikel, maschine, auftragsgroesse, vorgaenger_artikel, produkt_wechsel)
                predictions, prediction_made = self.get_predictions(input_data)

                alert = "Auftragsvorhersage erfolgreich!"
                return render_template('index.html',
                                    previous_inputs=request.form,
                                    predictions=predictions if prediction_made else None,
                                    product_codes_list=self.product_codes_list,
                                    line_codes_list=self.line_codes_list,
                                    prediction_made=prediction_made,
                                    alert=alert if prediction_made else None,
                                    compare_results=compare_results if prediction_made and form_type == 'compare_prediction' else None)

            elif form_type == 'compare_prediction':
                # Compare Prediction Logic
                artikel = request.form['artikel']
                artikel2 = request.form['artikel2']
                maschine = request.form['maschine']
                auftragsgroesse = request.form['auftragsgroesse']
                auftragsgroesse2 = request.form['auftragsgroesse2']
                vorgaenger_artikel = request.form['vorgaenger_artikel']
                produkt_wechsel = vorgaenger_artikel + '-' + artikel

                input_data = self.get_input_sample(artikel, maschine, auftragsgroesse, vorgaenger_artikel, produkt_wechsel)
                input_data2 = self.get_input_sample(artikel2, maschine, auftragsgroesse2, vorgaenger_artikel, produkt_wechsel)

                predictions, prediction_made = self.get_predictions(input_data)
                predictions2, prediction_made2 = self.get_predictions(input_data2)

                compare_results = self.do_prediction_comparison(predictions, predictions2)
                alert = "Auftragsvergleich erfolgreich!"

                return render_template('index.html',
                                    previous_inputs=request.form,
                                    predictions=predictions if prediction_made else None,
                                    predictions2=predictions2 if prediction_made2 else None,
                                    product_codes_list=self.product_codes_list,
                                    line_codes_list=self.line_codes_list,
                                    prediction_made=prediction_made,
                                    alert=alert if prediction_made else None,
                                    compare_results=compare_results if prediction_made and form_type == 'compare_prediction' else None)

            elif form_type == 'order_planning':
                order_data = request.form['order_data']
                maschine = request.form['maschine']
                vorgaenger_artikel = request.form['vorgaenger_artikel']
                calculation_method = request.form['calculation_method']

                # TODO IMPROVE USER DATA INPUT METHOD
                lines = order_data.strip().split('\n')
                data = [line.split() for line in lines]
                df_order_data = pd.DataFrame(data, columns=['ProductCode', 'Quantity'])
                df_order_data['Quantity'] = pd.to_numeric(df_order_data['Quantity'], errors='coerce')

                original_order = []
                current_prev_product = vorgaenger_artikel
                for index, row in df_order_data.iterrows():
                    cur_product = row['ProductCode']
                    cur_quantity = row['Quantity']
                    input_data = self.get_input_sample(cur_product, maschine, cur_quantity, current_prev_product, current_prev_product + '-' + cur_product)
                    predictions, prediction_made = self.get_predictions(input_data)
                    original_order.append({
                        'Position': index + 1,
                        'ProductCode': cur_product,
                        'Quantity': cur_quantity,
                        'OEE': predictions['OEE'],
                        'AVAIL': predictions['AVAIL'],
                        'PERF': predictions['PERF'],
                        'QUAL': predictions['QUAL'],
                        'APT': predictions['APT'],
                        'PBT': predictions['PBT']
                    })
                    current_prev_product = cur_product

                original_summary = {
                    'OEE': np.mean([order['OEE'] for order in original_order]),
                    'AVAIL': np.mean([order['AVAIL'] for order in original_order]),
                    'PERF': np.mean([order['PERF'] for order in original_order]),
                    'QUAL': np.mean([order['QUAL'] for order in original_order]),
                    'APT': np.sum([order['APT'] for order in original_order]),
                    'PBT': np.sum([order['PBT'] for order in original_order]),
                }

                original_order_dict = original_order

                alert = "Produktionsreihenfolge berechnet!"

                if calculation_method == 'greedy':
                    best_order = self.calculate_best_production_order_by_metric(df_order_data, 'OEE', vorgaenger_artikel, maschine)
                elif calculation_method == 'optimal' and len(df_order_data) <= 5:
                    best_order = self.calculate_optimal_production_order_by_metric(df_order_data, 'PBT', vorgaenger_artikel, maschine)
                elif calculation_method == 'sim_anneal':
                    best_order = self.calculate_optimal_production_order_by_sa(df_order_data, 'PBT', vorgaenger_artikel, maschine)

                if best_order is None or best_order.empty:
                    alert = "Anzahl an Elementen zu groß für optimale Suche!"
                else:
                    optimal_summary = {
                        'OEE': np.mean([order['OEE'] for order in best_order.to_dict('records')]),
                        'AVAIL': np.mean([order['AVAIL'] for order in best_order.to_dict('records')]),
                        'PERF': np.mean([order['PERF'] for order in best_order.to_dict('records')]),
                        'QUAL': np.mean([order['QUAL'] for order in best_order.to_dict('records')]),
                        'APT': np.sum([order['APT'] for order in best_order.to_dict('records')]),
                        'PBT': np.sum([order['PBT'] for order in best_order.to_dict('records')]),
                    }

                    best_order_dict = best_order.to_dict(orient='records')

                    # Combine best_order and original_order for comparison in template
                    combined_order = list(zip(best_order_dict, original_order_dict))

                return render_template('index.html', predictions=predictions,
                                    previous_inputs=request.form,
                                    product_codes_list=self.product_codes_list, 
                                    line_codes_list=self.line_codes_list,
                                    prediction_made=prediction_made, 
                                    alert=alert if prediction_made else None,
                                    compare_results=compare_results,
                                    best_order=best_order_dict if best_order_dict else None,
                                    original_order=original_order_dict,
                                    combined_order=combined_order,
                                    original_summary=original_summary,
                                    optimal_summary=optimal_summary if optimal_summary else None)

        return render_template('index.html', predictions=predictions,
                            product_codes_list=self.product_codes_list, line_codes_list=self.line_codes_list,
                            prediction_made=prediction_made, alert=alert,
                            compare_results=compare_results)

    def get_input_sample(self, artikel, maschine, auftragsgroesse, vorgaenger_artikel, produkt_wechsel):
        ## TODO MAKE THOSE DEFAULT VALUES ACTUALLY JUSTIFIED VALUES BY HISTORICAL STATISTICS
        # Define default values for cases where the change is not defined
        product_historic_times = self.latest_historic_times_feature[self.latest_historic_times_feature['ProductCode'] == artikel]
    
        default_auftragswechsel = product_historic_times['Historic_Avg_Auftragswechsel'].iloc[0]
        default_primaer = product_historic_times['Historic_Avg_Primär'].iloc[0]
        default_sekundaer = product_historic_times['Historic_Avg_Sekundär'].iloc[0]
        
        # Attempt to get the values or use default if not present
        auftragswechsel_value = self.latest_change_feature[self.latest_change_feature['ProductChange'] == produkt_wechsel]['10th_Percentile_Auftragswechsel']
        primaer_value = self.latest_change_feature[self.latest_change_feature['ProductChange'] == produkt_wechsel]['10th_Percentile_Primär']
        sekundaer_value = self.latest_change_feature[self.latest_change_feature['ProductChange'] == produkt_wechsel]['10th_Percentile_Sekundär']

        # Construct input dataframe 
        sample = pd.DataFrame({
            'ProductCode_encoded': [float(self.latest_product_enc[self.latest_product_enc['ProductCode']==artikel]['ProductCode_encoded'].iloc[0])],
            'Previous_ProductCode_encoded': [float(self.latest_product_enc[self.latest_product_enc['ProductCode']==vorgaenger_artikel]['ProductCode_encoded'].iloc[0])],
            'CALC_WIRKSTOFF_encoded': [float(self.latest_encoded_product_features[self.latest_encoded_product_features['ProductCode_encoded']==float(self.latest_product_enc[self.latest_product_enc['ProductCode']==artikel]['ProductCode_encoded'].iloc[0])]['CALC_WIRKSTOFF_encoded'].iloc[0])],
            'CALC_ALUFOLIE_encoded': [float(self.latest_encoded_product_features[self.latest_encoded_product_features['ProductCode_encoded']==float(self.latest_product_enc[self.latest_product_enc['ProductCode']==artikel]['ProductCode_encoded'].iloc[0])]['CALC_WIRKSTOFF_encoded'].iloc[0])],
            'Code_P-SARO_2': [float(self.latest_line_enc[self.latest_line_enc['Code']==maschine]['Code_P-SARO_2'].iloc[0])],
            'Code_V-LINIE6': [float(self.latest_line_enc[self.latest_line_enc['Code']==maschine]['Code_V-LINIE6'].iloc[0])],
            'Code_V-LINIE7': [float(self.latest_line_enc[self.latest_line_enc['Code']==maschine]['Code_V-LINIE7'].iloc[0])],
            'Code_V-LINIE8': [float(self.latest_line_enc[self.latest_line_enc['Code']==maschine]['Code_V-LINIE8'].iloc[0])],
            'Code_V-PAST-2': [float(self.latest_line_enc[self.latest_line_enc['Code']==maschine]['Code_V-PAST-2'].iloc[0])],
            'Code_V-TUBEN2': [float(self.latest_line_enc[self.latest_line_enc['Code']==maschine]['Code_V-TUBEN2'].iloc[0])],
            'OrderQuantity': [float(auftragsgroesse)],
            'FS_Breite': [float(self.latest_product_ref[self.latest_product_ref['ProductCode']==artikel]['FS_Breite'].iloc[0])],
            'FS_Länge': [float(self.latest_product_ref[self.latest_product_ref['ProductCode']==artikel]['FS_Länge'].iloc[0])],
            'FS_Tiefe': [float(self.latest_product_ref[self.latest_product_ref['ProductCode']==artikel]['FS_Tiefe'].iloc[0])],
            'PBL_Breite': [float(self.latest_product_ref[self.latest_product_ref['ProductCode']==artikel]['PBL_Breite'].iloc[0])],
            'PBL_Länge': [float(self.latest_product_ref[self.latest_product_ref['ProductCode']==artikel]['PBL_Länge'].iloc[0])],
            'Tuben_Durchmesser': [float(self.latest_product_ref[self.latest_product_ref['ProductCode']==artikel]['Tuben_Durchmesser'].iloc[0])],
            'CALC_PACKGROESSE': [float(self.latest_product_ref[self.latest_product_ref['ProductCode']==artikel]['CALC_PACKGROESSE'].iloc[0])],
            'Tuben_Länge': [float(self.latest_product_ref[self.latest_product_ref['ProductCode']==artikel]['Tuben_Länge'].iloc[0])],
            '10th_Percentile_Auftragswechsel': [float(auftragswechsel_value.iloc[0]) if not auftragswechsel_value.empty else default_auftragswechsel],
            '10th_Percentile_Primär': [float(primaer_value.iloc[0]) if not primaer_value.empty else default_primaer],
            '10th_Percentile_Sekundär': [float(sekundaer_value.iloc[0]) if not sekundaer_value.empty else default_sekundaer]

        })

        return sample

    def get_predictions(self, input_data):
        predictions = {}
        for target, model in self.models.items():
            if model:
                predictions[target] = model.predict(input_data)[0]
            else:
                predictions[target] = 'Model not found'

            if target in ['OEE', 'AVAIL', 'QUAL', 'PERF']:
                predictions[target] *= 100

        return predictions, True

    def do_prediction_comparison(self, pred1, pred2):
        res = {}
        # Check if both predictions have the same targets
        if pred1.keys() == pred2.keys():
            # Calculate the difference for each target
            for target in pred1:
                res[target] = pred2[target] - pred1[target]
        else:
            raise ValueError("Predictions do not have the same set of targets")

        return res
    
    def calculate_best_production_order_by_metric(self, order_data, metric, prev_product, production_line):
        columns = ['Position', 'ProductCode', 'Quantity', 'OEE', 'AVAIL', 'PERF', 'QUAL', 'APT', 'PBT']
        best_production_order = pd.DataFrame(columns=columns)
        new_prev_product = None
        new_order_data = order_data.copy()
        best_metric_result = 0
        best_position = 1
        best_order = None
        best_order_index = None

        while len(best_production_order) < len(order_data):
            ## For all still available orders
            for ord_ind in range(len(new_order_data)):
                cur_row = new_order_data.iloc[ord_ind]  # Using .iloc to properly access the DataFrame row
                cur_product = cur_row['ProductCode']
                cur_quantity = cur_row['Quantity']

                ## If first iteration use input field for prev product
                if new_prev_product == None:
                    input_data = self.get_input_sample(cur_product, \
                                                production_line, \
                                                cur_quantity, \
                                                prev_product, \
                                                prev_product+'-'+cur_product)
                else: # else use last found product in order
                    input_data = self.get_input_sample(cur_product, \
                                                production_line, \
                                                cur_quantity, \
                                                new_prev_product, \
                                                new_prev_product+'-'+cur_product)
                
                predictions, prediction_made = self.get_predictions(input_data)

                print("Iteration "+ str(best_position)+": Predicted " + str(cur_product) + " with Quantity " + str(cur_quantity) + " => Result=" + str(predictions['OEE']))

                ## If this order is better then the previously best found one, save it
                if predictions[metric] > best_metric_result:
                    best_metric_result = predictions[metric]
                    
                    best_order_index = ord_ind
                    best_order = {
                        'Position': best_position,
                        'ProductCode': cur_product,
                        'Quantity': cur_quantity,
                        'OEE':      predictions['OEE'],
                        'AVAIL':    predictions['AVAIL'],
                        'PERF':     predictions['PERF'],
                        'QUAL':     predictions['QUAL'],
                        'APT':      predictions['APT'],
                        'PBT':      predictions['PBT']                    
                    }

            ## Take the best found order
            best_production_order = best_production_order.append(best_order, ignore_index=True)
            new_prev_product = best_order['ProductCode']
            best_position += 1

            ## Remove best order from set of orders
            new_order_data = new_order_data.drop(index=best_order_index).reset_index(drop=True)

            ## Reset search function variables
            best_metric_result = 0
            best_order = None
            best_order_index = None

            print("\nIteration finished\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

        return best_production_order

    def calculate_optimal_production_order_by_metric(self, order_data, metric, prev_product, production_line):
        columns = ['Position', 'ProductCode', 'Quantity', 'OEE', 'AVAIL', 'PERF', 'QUAL', 'APT', 'PBT']
        optimal_production_order = pd.DataFrame(columns=columns)
        best_metric_sum = 1000000000
        best_order_sequence = None

        print("Started Calculation of optimal order for: ", order_data)
        all_permutations = list(itertools.permutations(order_data.index))
        permutation_count = len(all_permutations)
        print(f"Total permutations to evaluate: {permutation_count}")
        
        for perm_index, perm in enumerate(all_permutations):
            current_metric_sum = 0
            current_prev_product = prev_product
            current_order_sequence = []
            
            print(f"Evaluating permutation {perm_index + 1}/{permutation_count}: {perm}")

            for position, index in enumerate(perm):
                cur_row = order_data.loc[index]
                cur_product = cur_row['ProductCode']
                cur_quantity = cur_row['Quantity']
                
                input_data = self.get_input_sample(cur_product, production_line, cur_quantity, current_prev_product, current_prev_product + '-' + cur_product)
                predictions, prediction_made = self.get_predictions(input_data)

                print(f"  Position {position + 1}: Predicted {cur_product} with Quantity {cur_quantity} => {metric}={predictions[metric]}")

                current_metric_sum += predictions[metric]

                current_order_sequence.append({
                    'Position': position + 1,
                    'ProductCode': cur_product,
                    'Quantity': cur_quantity,
                    'OEE': predictions['OEE'],
                    'AVAIL': predictions['AVAIL'],
                    'PERF': predictions['PERF'],
                    'QUAL': predictions['QUAL'],
                    'APT': predictions['APT'],
                    'PBT': predictions['PBT']
                })

                current_prev_product = cur_product
            
            print(f"  Total {metric} for this permutation: {current_metric_sum}")

            if current_metric_sum < best_metric_sum:
                best_metric_sum = current_metric_sum
                best_order_sequence = current_order_sequence
                print(f"  New best order sequence found with total {metric}={best_metric_sum}")

        if best_order_sequence:
            optimal_production_order = pd.DataFrame(best_order_sequence, columns=columns)

        print("\nOptimal production order found:")
        print(optimal_production_order)
        return optimal_production_order

    # Function to evaluate the fitness of an individual
    def evaluate(self, individual, order_data, metric, prev_product, production_line):
        current_prev_product = prev_product
        total_metric = 0

        for index in individual:
            cur_row = order_data.iloc[index]
            cur_product = cur_row['ProductCode']
            cur_quantity = cur_row['Quantity']

            input_data = self.get_input_sample(cur_product, production_line, cur_quantity, current_prev_product, current_prev_product + '-' + cur_product)
            predictions, prediction_made = self.get_predictions(input_data)

            total_metric += predictions[metric]
            current_prev_product = cur_product

        return total_metric

    def calculate_optimal_production_order_by_sa(self, order_data, metric, prev_product, production_line, initial_temp=100, cooling_rate=0.95, num_iterations=500, early_stopping_iter=100):
        # Initial solution (random permutation of indices)
        current_solution = list(range(len(order_data)))
        random.shuffle(current_solution)
        current_cost = self.evaluate(current_solution, order_data, metric, prev_product, production_line)

        best_solution = current_solution[:]
        best_cost = current_cost

        temp = initial_temp
        no_improvement_counter = 0

        for iteration in range(num_iterations):
            # Create a neighbor solution by swapping two indices
            neighbor_solution = current_solution[:]
            i, j = random.sample(range(len(order_data)), 2)
            neighbor_solution[i], neighbor_solution[j] = neighbor_solution[j], neighbor_solution[i]
            neighbor_cost = self.evaluate(neighbor_solution, order_data, metric, prev_product, production_line)

            # Acceptance probability
            if neighbor_cost < current_cost or random.random() < np.exp((current_cost - neighbor_cost) / temp):
                current_solution = neighbor_solution
                current_cost = neighbor_cost

                # Update the best solution found so far
                if current_cost < best_cost:
                    best_solution = current_solution
                    best_cost = current_cost
                    no_improvement_counter = 0  # Reset counter if improvement found
                else:
                    no_improvement_counter += 1
            else:
                no_improvement_counter += 1

            # Print debugging information
            if iteration % 25 == 0:
                print(f"Iteration {iteration}: Best cost = {best_cost}, Current cost = {current_cost}, No Improvement Counter = {no_improvement_counter}")

            # Early stopping condition
            if no_improvement_counter >= early_stopping_iter:
                print(f"Stopping early at iteration {iteration} due to no improvement for {early_stopping_iter} iterations.")
                break

            # Cool down the temperature
            temp *= cooling_rate

        # Convert best solution indices back to order details
        best_order_sequence = []

        print("\nBest individual found:")
        for position, index in enumerate(best_solution):
            cur_row = order_data.iloc[index]
            input_data = self.get_input_sample(cur_row['ProductCode'], production_line, cur_row['Quantity'], prev_product, prev_product + '-' + cur_row['ProductCode'])
            predictions, prediction_made = self.get_predictions(input_data)

            best_order_sequence.append({
                'Position': position + 1,
                'ProductCode': cur_row['ProductCode'],
                'Quantity': cur_row['Quantity'],
                'OEE': predictions['OEE'],
                'AVAIL': predictions['AVAIL'],
                'PERF': predictions['PERF'],
                'QUAL': predictions['QUAL'],
                'APT': predictions['APT'],
                'PBT': predictions['PBT']
            })

            print(f"  Position {position + 1}: Product {cur_row['ProductCode']} with Quantity {cur_row['Quantity']}, OEE={predictions['OEE']}, AVAIL={predictions['AVAIL']}, PERF={predictions['PERF']}, QUAL={predictions['QUAL']}, APT={predictions['APT']}, PBT={predictions['PBT']}")

            prev_product = cur_row['ProductCode']

        return pd.DataFrame(best_order_sequence)