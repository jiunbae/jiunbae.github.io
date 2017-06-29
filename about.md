---
title: About
permalink: about/
profile: true
---

## Skills
<hr>
{% for skill in site.data.about.skills %}

<div class="skillbar clearfix " data-percent="{{ skill.performance }}0%">
    <div class="skillbar-title" style="background: {{ site.data.colors[skill.color] }}"><span>{{ skill.name }}</span></div>
    <div class="skillbar-bar" style="background: {{ site.data.colors[skill.color] }}"></div>
    <div class="skill-bar-description">{{ skill.description }}</div>
    <div class="skill-bar-percent">{{ skill.performance }}</div>
</div>
<div class="skill-description">
    {{ skill.experience }}
</div>

{% endfor %}

{% for about_item in site.data.about.info %}

## {{ about_item.label }}
<hr>

<div class="about items">
{% for item in about_item.data %}

<div class="about item about-{{ item.major }}">
    {% if item.major %}
        {% if item.link %}
            <a href="{{ item.link }}">
        {% endif %}

        <span class="about major">{{ item.major }}</span>

        {% if item.link %}
            </a>
        {% endif %}
    {% endif %}
        <div class="about minor">
            <span>{{ item.minor }}</span> <i>{{ item.sub }}</i>
        </div>
</div>


{% endfor %}
</div>

{% endfor %}


## [Resume]({{ site.resume }})

### Site's stack

Built with [Jekyll](http://jekyllrb.com/).  
Hosted on [Github Pages](https://pages.github.com/).  
Based on [Kactus](https://github.com/nickbalestra/kactus) theme.

{% include footer.html %}