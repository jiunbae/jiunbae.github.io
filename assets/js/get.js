$.get('https://github.com/MaybeS', function(data) {
    var graph = $('svg.js-calendar-graph-svg', $(data));
    var text = $('.js-contribution-graph h2', $(data));
    $('#timeline svg.calendar-graph-svg').html(graph);
    $('#timeline h4').html(text.text());

    var pinned = $('.pinned-repo-item', $(data));
});

window.twttr = (function(d, s, id) {
  var js, fjs = d.getElementsByTagName(s)[0],
    t = window.twttr || {};
  if (d.getElementById(id)) return t;
  js = d.createElement(s);
  js.id = id;
  js.src = "https://platform.twitter.com/widgets.js";
  fjs.parentNode.insertBefore(js, fjs);

  t._e = [];
  t.ready = function(f) {
    t._e.push(f);
  };
  return t;
}(document, "script", "twitter-wjs"));
