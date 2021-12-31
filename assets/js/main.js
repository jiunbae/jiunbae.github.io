// To make images retina, add a class "2x" to the img element
// and add a <image-name>@2x.png image. Assumes jquery is loaded.
function isRetina() {
	var mediaQuery = "(-webkit-min-device-pixel-ratio: 1.5),\
					  (min--moz-device-pixel-ratio: 1.5),\
					  (-o-min-device-pixel-ratio: 3/2),\
					  (min-resolution: 1.5dppx)";
	if (window.devicePixelRatio > 1)
		return true;
	if (window.matchMedia && window.matchMedia(mediaQuery).matches)
		return true;
	return false;
};
 
function retina() {
	if (!isRetina())
		return;
	$("img.2x").map(function(i, image) {
		var path = $(image).attr("src");
		path = path.replace(".png", "@2x.png");
		path = path.replace(".jpg", "@2x.jpg");
		$(image).attr("src", path);
	});
};

$(document).ready(retina);

// Navigation bar shrink effect
$(window).scroll(function() {
    if ($(document).scrollTop() > $('#front-cover').height() * 0.9) {
        $('nav').addClass('shrink');
    } else {
        $('#front-cover .cover-center').css('top', ($('#front-cover').height() + $(document).scrollTop()) * 0.5);
        $('nav').removeClass('shrink');
    }
});

// progress bar
jQuery('.skillbar').each(function(){
    jQuery(this).find('.skillbar-bar').animate({
        width:jQuery(this).attr('data-percent')
    },2000);
});

 var TxtType = function(el, toRotate, period) {
    this.toRotate = toRotate;
    this.el = el;
    this.loopNum = 0;
    this.period = parseInt(period, 10) || 2000;
    this.txt = '';
    this.tick();
    this.isDeleting = false;
};

TxtType.prototype.tick = function() {
    var fullTxt = this.toRotate[this.loopNum % this.toRotate.length];

    this.txt = fullTxt.substring(0, this.txt.length + ( this.isDeleting ? -1 : 1 ));

    this.el.innerHTML = '<span class="wrap">' + this.txt + '</span>';

    var that = this;
    var delta = 200 - Math.random() * 100;

    if (this.isDeleting) { delta /= 2; }

    if (!this.isDeleting && this.txt === fullTxt) {
        delta = this.period;
        if (this.toRotate.length > 1) {
            this.isDeleting = true;
        }
    } else if (this.isDeleting && this.txt === '') {
        this.isDeleting = false;
        this.loopNum++;
        delta = 500;
    }

    setTimeout(function() {
        that.tick();
    }, delta);
};

window.onload = function() {
    var elements = document.getElementsByClassName('typewrite');
    for (var i = 0; i < elements.length; i++) {
        var toRotate = elements[i].getAttribute('data-type');
        var period = elements[i].getAttribute('data-period');
        if (toRotate) {
            new TxtType(elements[i], toRotate.split(',') , period);
        }
    }
};
