const NUM_SAMPLES = 120;
const SAVE_FILENAME = "tests.json";

const TT_STATE = {
    IDLE: 1,
    GENERATING: 2,
    TESTING: 3
};


class TuringTest {
    constructor(num_samples) {
        this.num_samples = num_samples;
        this.tests = [];
        this.state = TT_STATE.IDLE;
        this.current_test = {
            start_time: -1,
            img_id: -1,
            correct_choice: -1
        };
    }

    genTest() {
        function get_img_src(img_id, is_ground_truth) {
            let src = "/images/test_" + img_id + "/";
            if (is_ground_truth)
                src += "true_color";
            else
                src += "after_train"
            return src + ".png";
        }

        this.state = TT_STATE.GENERATING;

        let img_id = Math.floor(Math.random() * this.num_samples);
        let show_truth = (Math.floor(Math.random() * 2) + 1) === 1;

        $("#test_img").attr('src', get_img_src(img_id, show_truth));

        this.current_test = {
            start_time: $.now(),
            img_id: img_id,
            is_truth: show_truth
        };

        this.state = TT_STATE.TESTING;
    }

    pickChoice(choice) {
        if (this.state === TT_STATE.TESTING) {
            let click_time = $.now();
            this.state = TT_STATE.IDLE;

            this.tests.push({
                img_id: this.current_test.img_id,
                is_truth: this.current_test.is_truth,
                correct: this.current_test.is_truth === choice,
                time: click_time - this.current_test.start_time
            });
        }
    }

    save(filename) {
        function download(content, fileName, contentType) {
            var a = document.createElement("a");
            var file = new Blob([content], {type: contentType});
            a.href = URL.createObjectURL(file);
            a.download = fileName;
            a.click();
        }
        let jsonData = JSON.stringify({tests: this.tests,
            num_samples: this.num_samples,
            generated: $.now()
        });
        download(jsonData, filename, 'text/plain');
    }
}

var turing_test = new TuringTest(NUM_SAMPLES);

$(document).ready(function () {
    $(window).keydown(function(e) {
        let key = e.which;

        if (key === 84) { // T
            $("#start-instruction").remove();
            turing_test.pickChoice(true);
            turing_test.genTest();
        }
        else if (key === 70) { // F
            $("#start-instruction").remove();
            turing_test.pickChoice(false);
            turing_test.genTest();
        }
        else if (key === 83) {
            turing_test.save(SAVE_FILENAME);
        }
    });
});
