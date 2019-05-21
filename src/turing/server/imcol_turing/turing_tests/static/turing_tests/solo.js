const NUM_SAMPLES = 120;

const TT_STATE = {
    IDLE: 1,
    GENERATING: 2,
    TESTING: 3
};


class TuringTest {
    constructor(num_samples) {
        this.num_samples = num_samples;
        this.num_tests = 0;
        this.state = TT_STATE.IDLE;
        this.current_test = {
            start_time: -1,
            img_id: -1,
            set_id: -1,
            correct_choice: -1
        };
    }

    genTest() {
        function get_img_src(img_id, is_ground_truth, set_id) {
            let set = "";
            switch (set_id) {
                case 1:
                    set = "unet";
                    break;
                case 2:
                    set = "gan";
                    break;
                case 3:
                    set = "vacgan";
                    break;
            }
            let src = "/static/turing_tests/images/" + set + "/test_" + img_id + "/";
            if (is_ground_truth)
                src += "true_color";
            else
                src += "after_train"
            return src + ".png";
        }

        let img_id = Math.floor(Math.random() * this.num_samples);
        let show_truth = (Math.floor(Math.random() * 2) + 1) === 1;
        let set_id = (Math.floor(Math.random() * 3) + 1);

        $("#test_img").attr('src', get_img_src(img_id, show_truth, set_id));

        this.current_test = {
            start_time: $.now(),
            img_id: img_id,
            set_id: set_id,
            is_truth: show_truth
        };

        this.state = TT_STATE.TESTING;
    }

    pickChoice(choice) {
        if (this.state === TT_STATE.TESTING) {
            let click_time = $.now();
            this.state = TT_STATE.GENERATING;

            let this_tt = this;

            $.ajax({
                type: "POST",
                url: URL_AJAX_SUBMIT,
                data: { csrfmiddlewaretoken: CSRF_TOKEN,
                        i: this.current_test.img_id,
                        si: this.current_test.set_id,
                        it: this.current_test.is_truth,
                        ic: this.current_test.is_truth === choice,
                        t: click_time - this.current_test.start_time
                      },
                success: function(data) {
                    if (data === "0") {
                        this_tt.genTest();
                        this_tt.num_tests++;
                        $("#num_tests").text(this_tt.num_tests);
                    }
                    else if (data === "1") {
                        this_tt.genTest();
                    }
                },
                error: function(request, status, error) {
                    alert("Unknown error");
                }
            });
        }
        else if (this.state === TT_STATE.IDLE) {
            this.genTest();
        }
    }
}

const turing_test = new TuringTest(NUM_SAMPLES);

$(document).ready(function () {
    $(window).keydown(function(e) {
        let key = e.which;

        if (key === 84) { // T
            $("#start-instruction").remove();
            turing_test.pickChoice(true);
        }
        else if (key === 70) { // F
            $("#start-instruction").remove();
            turing_test.pickChoice(false);
        }
    });
});
