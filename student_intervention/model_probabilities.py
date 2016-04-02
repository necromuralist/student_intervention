from sklearn.preprocessing import MinMaxScaler
from bokeh.palettes import Spectral7
tools = "pan,wheel_zoom,box_zoom,reset,resize,hover"
#tooltips = [('Variable', '@{0}'.format(variable)) for variable in non_zero_variables]
fig = b_figure(tools=tools, title='Variable Probabilities')
scaler = MinMaxScaler()
#x_normed = normalize(X_test)
for v_index, variable in enumerate(non_zero_variables):
    index = numpy.where(X_test.columns==variable)
    x_input = numpy.zeros((1, len(X_test.columns)))
    x = numpy.array([value for value in sorted(X_test[variable].unique())])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        x = scaler.fit_transform(x)
    y = []
    x = [x[0], x[-1]]
    for value in x:
        x_input[0][index] = value
        y.append(grid.best_estimator_.predict_proba(x_input)[0][1])
    #lines = plot.plot(x, y, label=variable)
    source = ColumnDataSource({variable: x, 'passed': y, 'name': [variable for item in x]})
    
    #fig.circle(variable, 'passed', name=variable, source=source)
    hover = fig.select(dict(type=HoverTool))
    hover.tooltips = [('Variable', '@name'), ("(x, y)", '($x, $y)')]
    hover.mode = 'mouse'
    #hover.line_policy='none'
    fig.line(x, y, source=source, line_color=Spectral7[v_index], legend=variable)
