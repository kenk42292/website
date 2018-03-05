/* global $, document */

/* Import header on to page */
$(document).ready(function() {
    'use strict';
    $(function() {
        $('.header-wrapper').load('/content/widgets/header.html');
    })
})

/* Make navigation item 'active' on click */
$(document).ready(function() {
    'use strict';
    $('.navbar-nav li a').click(function(event) { //eslint-disable-line no-unused-vars
        'use strict';
        // remove 'active' from all li
        $('.navbar-nav li a').parent().removeClass("active");
        // add 'active' to specific li
        $(this).parent().addClass("active");
    })
})

/* Extension of contains jquery selector to be case-insensitve */
$.expr[":"].containsi = $.expr.createPseudo(function(arg) {
    return function( elem ) {
        return $(elem).text().toUpperCase().indexOf(arg.toUpperCase()) >= 0;
    };
});

