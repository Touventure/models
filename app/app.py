from flask import Flask, request, jsonify
import pickle
import numpy as np
import tensorflow as tf
from content_based_modules import *
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from nearest_based import KNN
import os

app = Flask(__name__)
content_based_model = tf.keras.models.load_model('content_based.h5')
nearest_based_model = pickle.load(open("nearest_based.pkl", "rb"))


@app.route('/nearest_predict', methods=['POST'])
def nearest_predict():
    data = request.get_json()
    lat = data["lat"]
    long = data["long"]
    user_loc = tf.constant([[lat, long]], dtype=tf.float32)
    predictions = nearest_based_model.predict(user_loc).numpy().tolist()[0]
    predictions = [[place_id.decode(), place_name.decode(), city.decode()] for place_id, place_name, city in
                   predictions]
    return jsonify(predictions)


@app.route('/category_predict', methods=['POST'])
def genre_predict():
    new_data = request.get_json()
    new_id = new_data["id"]
    new_rating_count = new_data["rating_count"]
    new_rating_ave = new_data["rating_ave"]
    new_bahari = new_data["bahari"]
    new_budaya = new_data["budaya"]
    new_cagar_alam = new_data["cagar_alam"]
    new_pusat_perbelanjaan = new_data["pusat_perbelanjaan"]
    new_taman_hiburan = new_data["taman_hiburan"]
    new_tempat_ibadah = new_data["tempat_ibadah"]
    user_vec = np.array([[new_id, new_rating_count, new_rating_ave, new_bahari, new_budaya, new_cagar_alam,
                          new_pusat_perbelanjaan, new_taman_hiburan, new_tempat_ibadah]])

    item_vecs, destination_dict, item_vecs, item_train, user_train, y_train = load_data()
    scalerUser = StandardScaler()
    scalerUser.fit(user_train)
    scalerItem = StandardScaler()
    scalerItem.fit(item_train)
    scalerTarget = MinMaxScaler((-1, 1))
    scalerTarget.fit(y_train.reshape(-1, 1))

    u_s = 3
    i_s = 1

    user_vecs = gen_user_vecs(user_vec, len(item_vecs))
    suser_vecs = scalerUser.transform(user_vecs)
    sitem_vecs = scalerItem.transform(item_vecs)

    y_p = content_based_model.predict([suser_vecs[:, u_s:], sitem_vecs[:, i_s:]])
    y_pu = scalerTarget.inverse_transform(y_p)

    sorted_index = np.argsort(-y_pu, axis=0).reshape(-1).tolist()
    sorted_ypu = y_pu[sorted_index]
    sorted_items = item_vecs[sorted_index]
    predicts = get_predict(sorted_ypu, sorted_items, destination_dict, maxcount=10)
    return predicts


if __name__ == '__main__':
    # app.run(port=5000, debug=True)
    server_port = os.environ.get('PORT', '8080')
    app.run(debug=False, port=server_port, host='0.0.0.0')
