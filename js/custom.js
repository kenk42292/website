/* global $, document */
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
