
function configNavbar() {

    d3.select("#sideNavbar")
      .on("mouseover", function(){
            d3.select("#sideNavbar")
                //.style("background-color", 'red')
                //.style("width", 200+'px');
                .attr("class", "navbarExpanded");
      })
      .on("mouseout", function(){
            d3.select("#sideNavbar")
                //.style("background-color", 'black')
                //.style("width", 50+'px');
                .attr("class", "navbarCollapsed");
      });

      /*
    d3.select("#"+navbarActive)
        .attr("class", "active");

    var linkNames = ["#dashLink", "#pregLink"];
    */

    d3.select("#dashLink")
        .on("click", function() {
            d3.selectAll(".active").attr("class", "inactive");
            d3.select("#dashLink").attr("class", "active");
        });

    d3.select("#pregLink")
        .on("click", function() {
            d3.selectAll(".active").attr("class", "inactive");
            d3.select("#pregLink").attr("class", "active");
        });
}



function makeDashPanel(options) {

    var title = options.title,
        body = options.body,
        dom = options.dom,
        w = options.w  //options.w || d3.select(dom).node().clientWidth,
        h = options.h  //options.h || d3.select(dom).node().clientHeight,
        c = options.c,
        id = options.id;
    
    var svgP = d3.select(dom).append("svg")
        .attr("class", "dashPanel")
        .attr("id", "dashSvg"+id)
        .attr("width", w)
        .attr("height", h);
    
    svgP.append("text") // title, small font
        .text(title)
        .attr("class", "dashTitle"+c)
        .attr("x", (w/ 2))
        .attr("y", 10);
    
    svgP.append("text") // body, large font
        .text(body)
        .attr("class", "dashBody"+c)
        .attr("x", w / 2)
        .attr("y", (h / 2)); 
        
};


function appendDashPanelLine(options) {

    var svg = options.svg,
    	dom = options.dom,
        df = options.df,
        xcol = options.xcol,
        ycol = options.ycol,
        w = options.w
        h = options.h
        pad = options.pad;

    var x = d3.time.scale()  // x scale
        .domain([d3.min(df, function(d) {return d[xcol]}), d3.max(df, function(d) {return d[xcol]})])
        .range([pad, w-pad]);

    var y = d3.scale.linear()   // y scale
        .domain([d3.min(df, function(d) {return d[ycol]}), d3.max(df, function(d) {return d[ycol]})])
        .range([h-pad, pad]); 
     
    // line
    var line = d3.svg.line()
        .x(function(d) {return x(d[xcol])})
        .y(function(d) {return y(d[ycol])});
        
    svg.append("path")
          .datum(df)
          .attr("class", "line")
          //.attr("clip-path", "url(#clip)")
          .attr("d", line)
          .style("fill", "none")
          .style("stroke-width", "4")
          .style("stroke", "#1b9e77"); 

    // points
    svg.selectAll(".point")
        .data(df)
        .enter().append("circle")
        //.attr("clip-path", "url(#clip)")
        .attr("r", 4)
        .attr("cx", function(d) { return x(d[xcol]); })
        .attr("cy", function(d) { return y(d[ycol]); })
        .style("fill", "white")
        .style("stroke","#1b9e77");
        //.style("opacity", ptFillOp)
        //.style("stroke-opacity", ptStrOp);
}


///////////////////
//
//    dashboard constructor
//
///////////////////

function makeDashChart() {
    wD = d3.select("#containerD").node().clientWidth- dashMarg.right - dashMarg.left;
    hD = d3.select("#containerD").node().clientHeight- dashMarg.top - dashMarg.bottom;

    svgDash = d3.select("#containerD")
                .append("svg")
                .attr("width", wD + dashMarg.right + dashMarg.left)
                .attr("height", hD + dashMarg.top + dashMarg.bottom)
            .append("g")
                .attr("transform", "translate(" + dashMarg.left + "," + dashMarg.top + ")");

    svgDash.append("text")
        .text("Segment Explorer")
        .attr("class", "chartTitle");
}

function makeDash(options) {
    // takes in following values, then attaches dashboard to provided svg.
    // titles.length should be divisible by numRows.

    var //dom = options.dom,
        svg = svgDash//options.svg,
        titles = options.titles,
        values = options.values,
        numRows = options.numRows,
        dashUnits = options.units;

                /*
    svg.append("text")
        .text("Panel")
        .attr("class", "chartTitle");*/

    b0 = svgDash.selectAll("g")  
                .data(values)
                .enter().append("g")
                .attr("transform", makeSpacing);
                       
    b0.append("text")
        .attr("x", 0)
        .attr("y", 60)
        .attr("class", "dashboardUnit")
        .style("text-anchor", "middle")
        .text(function(d, i) { return dashUnits[i]; });

    b0.append("text")
        .attr("x", 0)
        .attr("y", 45)
        .attr("class", "dashboardBody")
        .style("text-anchor", "middle")
        .text(function(d) { return d; });


    b0.append("text")
        .attr("x", 0)
        .attr("y", 20)
        .attr("class", "dashboardTitle")
        .style("text-anchor", "middle")
        .text(function(d, i) { return titles[i]; }); 


    function makeSpacing(d, i) {
        //var numRows = 2   // numRows and numCols must multiply to tt0.length
        var numCols = titles.length / numRows;

        return "translate(" + 
        (((i+numCols)%numCols) // normalizer (ie 0, 1, or 2 in this case)
        *(wD / numCols ) + //space btwn columns
        ((wD / numCols )/2) // using center-aligned text, so adding half column width
        ) 
        + ","+
        (Math.trunc(i/numCols)* // normalizer
        (hD / numRows)) //space btw rows
        +")"
    };
}


// Creates the histogram chart. Lays the foundation for placing and updating hist bars.
// Placing and updating hist bars will use variables created in this step.
function makeHistChart(options) {

    var dom = options.dom,
        margin = options.margin,
        values = options.values;

    hH = options.h - margin.top - margin.bottom,
    wH = options.w - margin.right - margin.left,
    numBins = options.numBins;

    // A formatter for counts.
    var formatCount = d3.format(",.0f");

    xH = d3.scale.linear()
        .domain([50, 190])
        .range([0, wH]);

    var xAxis = d3.svg.axis()
        .scale(xH)
        .orient("bottom")
        .outerTickSize(0);

    svgH = d3.select(dom).append("svg")
        .attr("width", wH + margin.left + margin.right)
        .attr("height", hH + margin.top + margin.bottom)
      .append("g")
        .attr("transform", "translate(" + margin.left + "," + margin.top + ")")
        .attr('id', "histSvg");

    svgH.append("g")
        .attr("class", "axis")
        .attr("transform", "translate(0," + hH + ")")
        .call(xAxis);

    svgH.append("text")
        .text("Histogram")
        .attr("class", "chartTitle");

    
    makeHistY = function(val) {
        // Generate a histogram using twenty uniformly-spaced bins.
        dataH = d3.layout.histogram()
            .bins(xH.ticks(numBins))
            (val);

        yH = d3.scale.linear()
            .domain([0, d3.max(dataH, function(d) { return d.y; })])
            .range([hH, 0]);

            /*
        var yAxis = d3.svg.axis()
            .scale(yH)
            .orient("left")
            .ticks(0)
            .outerTickSize(0);

        svgH.append("g")
            .attr("id", "yAxis")
            .attr("class", "y axis")
            //.attr("transform", "translate(0," + hH + ")")
            .call(yAxis);*/
    }
    makeHistY(values);

    makeHistBars = function (values) {
        var bar = svgH.selectAll(".bar")
            .data(dataH)
          .enter().append("g")
            .attr("id", "histBars")
            .attr("transform", function(d) { return "translate(" + xH(d.x) + "," + yH(d.y) + ")"; });

        bar.append("rect")
            .attr("x", 1)
            .attr("width", (wH/numBins)+2)
            .attr("height", function(d) { return hH - yH(d.y); })
            .style("fill", function(d) {return colorR(d.x)});
    };
    makeHistBars(values);   
}

// Hist chart must previously have been created. This just takes in an array and
// makes the hist bars based on it. Separated out for use in updating bars.
 
///////////////////
//
//    Legend
//
///////////////////

function addLegend() {        
    var color = d3.scale.linear()  // color scale
        .domain([3,2,1,0])
        .range(["#d7191c","#fdae61","#a6d96a","#1a9641"]);

    var legendRectSize = 7;                                  
    var legendSpacing = 7;  
    var xPlacement = 5;
    var yPlacement = 5;
    var legendText = ["Recovery", "Easy", "Stamina", "Impulse"];

    var wL = d3.select("#legendContainer").node().clientWidth,
        hL = d3.select("#legendContainer").node().clientHeight;

    var legendSvg = d3.select("#legendContainer")
                        .append("svg")
                        .attr("height", hL)
                        .attr("width", wL);

    var legend = legendSvg.selectAll('.legend')
                    .data(color.domain())
                    .enter()
                    .append("g")
                    .attr("class", "legend")
                    .attr("transform", function(d, i) {
                        var vert = i*(legendRectSize + legendSpacing)+yPlacement
                        return "translate("+xPlacement+','+ vert+")";
                        });
                    
    legend.append('rect')                                     
          .attr('width', legendRectSize)                          
          .attr('height', legendRectSize)                         
          .style('fill', color)                                   
          .style('stroke', color);      

    legend.append('text')                                     
          .attr('x', legendRectSize + legendSpacing)              
          .attr('y', legendRectSize )              
          .text(function(d) { return legendText[d]; });
}


/////////////////////////////////////////////////
//
//      Scales, colors, and specs
//
/////////////////////////////////////////////////

////////////////////////
//     scales
////////////////////////

// hr scale for use on map 
var hrColorRange =['#a50026','#d73027','#f46d43','#fdae61','#a6d96a', '#66bd63','#1a9850','#006837','#006837','#006837','#006837'].reverse()  

var colorR = d3.scale.linear()
    .domain([80,90,100,110,120,130,140,150,160,170,180]) // backwards bc range is backwards
    .range(hrColorRange);

var color = d3.scale.linear()  // color scale
      .domain([3,2,1,0])
      .range(["#d7191c","#fdae61","#a6d96a","#1a9641"]);


// for converting units into miles or kilometers
function u(x) {
    var km = x/1000;
    if (units=="miles") {
        return km / 1.6;
    }
}

function t(x) { //returns hours
    /*
    if (x>60*60) {return x/(60*60)} // if less than an hour, return minutes
    else {return x/(60)}; // if greater than an hour, return hours
*/
    return x/3600;
}

function uMaf(x) {
    var kmPerSec = x/1000;
    var kmPerHr = kmPerSec *60 * 60;
    if (units=="miles") {
        var mph = kmPerHr / 1.6;
        if (mph < 1) {mph = 2};
        var minPerMile = 60/mph;
        return minPerMile;
    }
}

////////////////////////
//     specs
////////////////////////

var masterMargin = {top: 30, right: 30, bottom: 30, left: 30};

var histMargin = masterMargin,
    dashMarg = {top: 30, right: 15, bottom: 15, left: 15},
    format = d3.time.format("%c");

// the common width for items along the side of the page
var cW = 300;






//////////////////////////////////////////////////////
//
//         Presets
//
///////////////////////////////////////////////////////

var barStacked = true; // boolean to help w stacked vs grouped state of bars
var units = "miles";
var speedMeasure = "min/mile";
  