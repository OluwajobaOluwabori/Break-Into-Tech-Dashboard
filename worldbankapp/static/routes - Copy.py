from worldbankapp import app

import json, plotly
from flask import render_template, request, Response, jsonify
from scripts.data import return_figures
from scripts.breakintotech import chart

@app.route('/', methods=['POST', 'GET'])
@app.route('/index', methods=['POST', 'GET'])
def index():

	# List of countries for filter
	roles = ['Data or business analyst',
'Data scientist or machine learning specialist',
'Developer, back-end',
'Developer, front-end',
'Developer, full-stack',
'DevOps specialist',
'Designer',
'Developer, embedded applications or devices',]

	# Parse the POST request countries list
	if (request.method == 'POST') and request.form:
		figures = return_figures(request.form)
		roles_selected = []

		for role in request.form.lists():
			roles_selected.append(role[1])
	
	# GET request returns all countries for initial page load
	else:
		figures = return_figures()
		roles_selected = []
		for role in roles:
			roles_selected.append(role[1])

	# plot ids for the html id tag
	ids = ['figure-{}'.format(i) for i, _ in enumerate(figures)]

	# Convert the plotly figures to JSON for javascript in html template
	figuresJSON = json.dumps(figures, cls=plotly.utils.PlotlyJSONEncoder)

	return render_template('data.html', ids=ids,
		figuresJSON=figuresJSON,
		all_countries=country_codes,
		countries_selected=countries_selected)

