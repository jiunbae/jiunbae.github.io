---
title: About
permalink: about/
profile: true
---

{% for about_item in site.data.about %}
# {{ about_item.label }}

{% for item in about_item.data %}
- {% if item.bold %}{% if item.link %}[{% endif %}**{{ item.bold }}**{% if item.link %}]({{ item.link }}){% endif %}{% endif %} {% if item.italic %}*{{ item.italic }}*{% endif %} {{ item.normal }}

{% endfor %}
<hr>
{% endfor %}


## [Resume]({{ site.resume }})

### Site's stack

Built with [Jekyll](http://jekyllrb.com/).  
Hosted on [Github Pages](https://pages.github.com/).  
Based on [Kactus](https://github.com/nickbalestra/kactus) theme.

{% include footer.html %}