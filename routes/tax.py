from flask import Blueprint, jsonify, render_template, request

import numpy as np

tax_bp = Blueprint('tax', __name__)

# Slabs (duplicated from original to preserve behavior)
NEW_REGIME_SLABS = [
	(400000, 0), (800000, 5), (1200000, 10), (1600000, 15), (2000000, 20), (2400000, 25), (float('inf'), 30)
]
OLD_REGIME_SLABS_UNDER_60 = [(250000, 0), (500000, 5), (1000000, 20), (float('inf'), 30)]
OLD_REGIME_SLABS_60_TO_80 = [(300000, 0), (500000, 5), (1000000, 20), (float('inf'), 30)]
OLD_REGIME_SLABS_80_PLUS = [(500000, 0), (1000000, 20), (float('inf'), 30)]


def compute_slab_tax(income, slabs):
	remaining = income
	lower = 0
	tax = 0
	steps = []
	for upto, rate in slabs:
		band_amt = max(0, min(remaining, upto - lower))
		if band_amt > 0:
			t = band_amt * rate / 100
			tax += t
			steps.append({'band': [lower + 1, upto], 'amount': band_amt, 'rate': rate, 'tax': t})
			remaining -= band_amt
			lower = upto
		if remaining <= 0:
			break
	return {'tax': tax, 'steps': steps}


def compute_surcharge(base_tax, total_income, regime):
	t = total_income
	rate = 0
	if t > 50000000 and regime == "old":
		rate = 37
	if t > 20000000:
		rate = max(rate, 25)
	if t > 10000000:
		rate = max(rate, 15)
	if t > 5000000:
		rate = max(rate, 10)
	if regime == "new" and rate == 37:
		rate = 25
	return {'rate': rate, 'surcharge': base_tax * rate / 100}


def apply_87a_rebate_new_regime(base_tax, taxable_excl_special):
	if taxable_excl_special <= 1200000:
		return max(0, base_tax - 60000)
	return base_tax


def calc_new_regime(inputs):
	std_ded = 75000
	family_pension_cap = 25000
	normal_income = max(0, inputs['salary_income'] + inputs['other_income'] - std_ded - min(inputs['family_pension_deduction'], family_pension_cap) - inputs['employer_nps_80ccd2'])
	slab_result = compute_slab_tax(normal_income, NEW_REGIME_SLABS)
	slab_tax = slab_result['tax']
	stcg_tax = inputs['stcg_111a'] * 0.15
	ltcg_eq_exemption = max(0, inputs['ltcg_112a'] - 100000)
	ltcg_eq_tax = ltcg_eq_exemption * 0.10
	other_ltcg_tax = inputs['other_ltcg_112'] * (inputs['other_ltcg_112_rate'] / 100)
	slab_tax_after_rebate = apply_87a_rebate_new_regime(slab_tax, normal_income)
	base_tax = slab_tax_after_rebate + stcg_tax + ltcg_eq_tax + other_ltcg_tax
	gross_total_income = (inputs['salary_income'] + inputs['other_income'] + inputs['stcg_111a'] + inputs['ltcg_112a'] + inputs['other_ltcg_112'])
	surcharge_result = compute_surcharge(base_tax, gross_total_income, "new")
	cess = 0.04 * (base_tax + surcharge_result['surcharge'])
	return {
		'regime': 'new',
		'normal_income': normal_income,
		'steps': slab_result['steps'],
		'slab_tax': slab_tax,
		'slab_tax_after_rebate': slab_tax_after_rebate,
		'stcg_tax': stcg_tax,
		'ltcg_eq_tax': ltcg_eq_tax,
		'other_ltcg_tax': other_ltcg_tax,
		'surcharge_rate': surcharge_result['rate'],
		'surcharge': surcharge_result['surcharge'],
		'cess': cess,
		'total_tax': max(0, base_tax + surcharge_result['surcharge'] + cess)
	}


def calc_old_regime(inputs):
	std_ded_old = 50000 if (inputs['old_std_ded'] and inputs['salary_income'] > 0) else 0
	cap_80c = min(inputs['ded_80c'], 150000)
	cap_80d = min(inputs['ded_80d'], 75000)
	cap_ccd1b = min(inputs['ded_80ccd1b'], 50000)
	hp_loss = -1 * min(inputs['housing_loan_interest'], 200000)
	other_old = max(0, inputs['other_deductions_old'])
	gross = (inputs['salary_income'] + inputs['other_income'] + inputs['stcg_111a'] + inputs['ltcg_112a'] + inputs['other_ltcg_112'])
	normal_gti = max(0, inputs['salary_income'] + inputs['other_income'] - std_ded_old + hp_loss)
	deductions = cap_80c + cap_80d + cap_ccd1b + other_old
	normal_income = max(0, normal_gti - deductions)
	if inputs['age'] == "<60":
		slabs = OLD_REGIME_SLABS_UNDER_60
	elif inputs['age'] == "60-80":
		slabs = OLD_REGIME_SLABS_60_TO_80
	else:
		slabs = OLD_REGIME_SLABS_80_PLUS
	slab_result = compute_slab_tax(normal_income, slabs)
	stcg_tax = inputs['stcg_111a'] * 0.15
	ltcg_eq_exemption = max(0, inputs['ltcg_112a'] - 100000)
	ltcg_eq_tax = ltcg_eq_exemption * 0.10
	other_ltcg_tax = inputs['other_ltcg_112'] * (inputs['other_ltcg_112_rate'] / 100)
	slab_tax_after_rebate = slab_result['tax']
	if (inputs['resident'] and normal_income <= 500000 and inputs['stcg_111a'] == 0 and inputs['ltcg_112a'] == 0 and inputs['other_ltcg_112'] == 0):
		slab_tax_after_rebate = max(0, slab_result['tax'] - 12500)
	base_tax = slab_tax_after_rebate + stcg_tax + ltcg_eq_tax + other_ltcg_tax
	surcharge_result = compute_surcharge(base_tax, gross, "old")
	cess = 0.04 * (base_tax + surcharge_result['surcharge'])
	return {
		'regime': 'old',
		'normal_income': normal_income,
		'steps': slab_result['steps'],
		'slab_tax': slab_result['tax'],
		'slab_tax_after_rebate': slab_tax_after_rebate,
		'stcg_tax': stcg_tax,
		'ltcg_eq_tax': ltcg_eq_tax,
		'other_ltcg_tax': other_ltcg_tax,
		'surcharge_rate': surcharge_result['rate'],
		'surcharge': surcharge_result['surcharge'],
		'cess': cess,
		'total_tax': max(0, base_tax + surcharge_result['surcharge'] + cess),
		'details': {
			'std_ded_old': std_ded_old,
			'cap_80c': cap_80c,
			'cap_80d': cap_80d,
			'cap_ccd1b': cap_ccd1b,
			'hp_loss': hp_loss,
			'other_old': other_old,
			'deductions': deductions,
		}
	}


def suggest_optimization(inputs):
	old_res = calc_old_regime(inputs)
	remaining_80c = max(0, 150000 - min(inputs['ded_80c'], 150000))
	remaining_ccd1b = max(0, 50000 - min(inputs['ded_80ccd1b'], 50000))
	budget = inputs['investment_budget']
	to_80c = min(remaining_80c, budget)
	to_ccd1b = min(remaining_ccd1b, budget - to_80c)
	total_add = to_80c + to_ccd1b
	# marginal rate estimate
	if inputs['age'] == "<60":
		slabs = OLD_REGIME_SLABS_UNDER_60
	elif inputs['age'] == "60-80":
		slabs = OLD_REGIME_SLABS_60_TO_80
	else:
		slabs = OLD_REGIME_SLABS_80_PLUS
	last_rate = 30
	for upto, rate in slabs:
		if old_res['normal_income'] <= upto:
			last_rate = rate
			break
	est_savings = total_add * last_rate / 100
	return {
		'to_80c': to_80c,
		'to_ccd1b': to_ccd1b,
		'total_add': total_add,
		'est_savings': est_savings,
		'last_rate': last_rate,
		'note': "Investing the suggested amounts under the old regime could lower taxable income; compare regimes after applying.",
	}


@tax_bp.route('/tax-calculator')
def tax_calculator():
	return render_template('tax_calculator.html')


@tax_bp.route('/calculate-tax', methods=['POST'])
def calculate_tax():
	try:
		data = request.get_json()
		inputs = {
			'resident': data.get('resident', True),
			'age': data.get('age', '<60'),
			'salary_income': float(data.get('salary_income', 0)),
			'other_income': float(data.get('other_income', 0)),
			'stcg_111a': float(data.get('stcg_111a', 0)),
			'ltcg_112a': float(data.get('ltcg_112a', 0)),
			'other_ltcg_112': float(data.get('other_ltcg_112', 0)),
			'other_ltcg_112_rate': float(data.get('other_ltcg_112_rate', 20)),
			'old_std_ded': data.get('old_std_ded', True),
			'ded_80c': float(data.get('ded_80c', 0)),
			'ded_80d': float(data.get('ded_80d', 0)),
			'ded_80ccd1b': float(data.get('ded_80ccd1b', 0)),
			'housing_loan_interest': float(data.get('housing_loan_interest', 0)),
			'other_deductions_old': float(data.get('other_deductions_old', 0)),
			'employer_nps_80ccd2': float(data.get('employer_nps_80ccd2', 0)),
			'family_pension_deduction': float(data.get('family_pension_deduction', 0)),
			'investment_budget': float(data.get('investment_budget', 0)),
		}
		new_regime_result = calc_new_regime(inputs)
		old_regime_result = calc_old_regime(inputs)
		better_regime = "new" if new_regime_result['total_tax'] <= old_regime_result['total_tax'] else "old"
		tax_difference = abs(new_regime_result['total_tax'] - old_regime_result['total_tax'])
		def clean_steps(steps):
			cleaned_steps = []
			for step in steps:
				cleaned_steps.append({'band': [step['band'][0] if step['band'][0] != float('inf') else 999999999, step['band'][1] if step['band'][1] != float('inf') else 999999999], 'amount': step['amount'], 'rate': step['rate'], 'tax': step['tax']})
			return cleaned_steps
		return jsonify({
			'success': True,
			'new_regime': {
				'total_tax': round(new_regime_result['total_tax'], 2),
				'normal_income': round(new_regime_result['normal_income'], 2),
				'slab_tax': round(new_regime_result['slab_tax'], 2),
				'slab_tax_after_rebate': round(new_regime_result['slab_tax_after_rebate'], 2),
				'stcg_tax': round(new_regime_result['stcg_tax'], 2),
				'ltcg_eq_tax': round(new_regime_result['ltcg_eq_tax'], 2),
				'other_ltcg_tax': round(new_regime_result['other_ltcg_tax'], 2),
				'surcharge_rate': new_regime_result['surcharge_rate'],
				'surcharge': round(new_regime_result['surcharge'], 2),
				'cess': round(new_regime_result['cess'], 2),
				'steps': clean_steps(new_regime_result['steps'])
			},
			'old_regime': {
				'total_tax': round(old_regime_result['total_tax'], 2),
				'normal_income': round(old_regime_result['normal_income'], 2),
				'slab_tax': round(old_regime_result['slab_tax'], 2),
				'slab_tax_after_rebate': round(old_regime_result['slab_tax_after_rebate'], 2),
				'stcg_tax': round(old_regime_result['stcg_tax'], 2),
				'ltcg_eq_tax': round(old_regime_result['ltcg_eq_tax'], 2),
				'other_ltcg_tax': round(old_regime_result['other_ltcg_tax'], 2),
				'surcharge_rate': old_regime_result['surcharge_rate'],
				'surcharge': round(old_regime_result['surcharge'], 2),
				'cess': round(old_regime_result['cess'], 2),
				'steps': clean_steps(old_regime_result['steps']),
				'deductions': old_regime_result['details']['deductions']
			},
			'comparison': {
				'better_regime': better_regime,
				'tax_difference': round(tax_difference, 2),
				'savings_with_better': round(tax_difference, 2)
			},
			'optimization': {
				'to_80c': round(suggest_optimization(inputs)['to_80c'], 2),
				'to_ccd1b': round(suggest_optimization(inputs)['to_ccd1b'], 2),
				'total_add': round(suggest_optimization(inputs)['total_add'], 2),
				'est_savings': round(suggest_optimization(inputs)['est_savings'], 2),
				'last_rate': suggest_optimization(inputs)['last_rate'],
				'note': suggest_optimization(inputs)['note']
			}
		})
	except Exception as e:
		return jsonify({'error': str(e)}), 500



