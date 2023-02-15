from worldbankapp import app

import json, plotly
from flask import render_template, request, Response, jsonify, Flask
#from scripts.data import return_figures
from scripts.breakintotech import chart
#from scripts.breakintotech import plot
import plotly.graph_objs as go
import plotly.colors

##undo
figures=chart() 
layout = dict(title = 'Change in Rural Population <br> (Percent of Total Population)')   
#figures=figure

ids = ["graph-{}".format(i) for i, _ in enumerate(figures)]
figuresJSON = json.dumps(figures, cls=plotly.utils.PlotlyJSONEncoder)


@app.route("/", methods=['GET'])
@app.route("/index", methods=['GET'])
def home():
	   # encode plotly graphs in JSON
#undo

	# render web page with plotly graphs
	return render_template('home.html', ids=ids, figuresJSON=figuresJSON)
#end undo

#plot=plot('LanguageHaveWorkedWith','LanguageWantToWorkWith')


#@app.route('/', methods=['POST', 'GET'])
# @app.route('/index', methods=['POST', 'GET'])
# def index():

# 	# List of countries for filter
# 	roles = ['Data or business analyst',
# 			'Data scientist or machine learning specialist',
# 			 'Developer, back-end'#,
# 			# 'Developer, front-end',
# 			# 'Developer, full-stack',
# 			# 'DevOps specialist',
# 			# 'Designer',
# 			# 'Developer, embedded applications or devices'
# 			]

# 	# Parse the POST request countries list
# 	if (request.method == 'POST') and request.form:
# 		figures = chart('LanguageHaveWorkedWith','LanguageWantToWorkWith',request.form)
# 		roles_selected = []

# 		for role in request.form.lists():
# 			roles_selected.append(role[1][0])
	
# 	# GET request returns all countries for initial page load
# 	else:
# 		figures = chart('LanguageHaveWorkedWith','LanguageWantToWorkWith')
# 		roles_selected = []
# 		for role in roles:
# 			roles_selected.append(role)

# 	# plot ids for the html id tag
# 	ids = ['figure-{}'.format(i) for i, _ in enumerate(figures)]

# 	# Convert the plotly figures to JSON for javascript in html template
# 	figuresJSON = json.dumps(figures, cls=plotly.utils.PlotlyJSONEncoder)

# 	return render_template('index.html', ids=ids,
# 		figuresJSON=figuresJSON,
# 		all_roles=roles,
# 		roles_selected=roles_selected)


#data=chart('LanguageHaveWorkedWith','LanguageWantToWorkWith')
# @app.route('/')
# def data():
# 	return render_template('home.html', ids=ids,
# 		figuresJSON=figuresJSON)














