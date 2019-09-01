(function($) {

	$(".navbar-collapse a").on('click', function() {
		$(".navbar-collapse.collapse").removeClass('in');
	});

	//jQuery to collapse the navbar on scroll
	$(window).scroll(function() {
		if ($(".navbar-default").offset().top > 50) {
			$(".navbar-fixed-top").addClass("top-nav-collapse");
		} else {
			$(".navbar-fixed-top").removeClass("top-nav-collapse");
		}
	});

})(jQuery);



2
function get_image() {

	$('#testcaptiondiv').text("");

	$('#originalscaptiondiv').text("");

	var img = document.getElementById("testim");


	if($('#dataset').val() === 'flickr'){

		img.src = "https://raw.githubusercontent.com/luca-ant/WhatsSee_dataset/master/images/"+$('#testimages').val()


	}
	else if($('#dataset').val() ==='coco'){
		img.src = "http://images.cocodataset.org/train2017/"+$('#testimages').val()




	}



}



