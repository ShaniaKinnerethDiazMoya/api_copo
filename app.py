import numpy as np
import matplotlib
# Usar backend Agg que no requiere GUI
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import io
import base64
from datetime import datetime
from flask import Flask, request, jsonify, send_file, render_template, redirect, url_for
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Configuración
UPLOAD_FOLDER = 'static/images'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def koch_snowflake_external_step(points, step, scale_factor=1.0):
    """Genera el copo de nieve completo con triángulos externos"""
    if step == 0:
        side_length = 1.0 * scale_factor
        height = (np.sqrt(3) / 2) * side_length
        p1 = np.array([0, 0])
        p2 = np.array([side_length, 0])
        p3 = np.array([side_length / 2, height])
        return np.array([p1, p2, p3, p1])
    
    new_points = []
    for i in range(len(points) - 1):
        p0 = points[i]
        p1_val = points[i + 1]
        segment = p1_val - p0
        one_third = p0 + segment / 3
        two_thirds = p0 + 2 * segment / 3
        
        rotation_matrix = np.array([[np.cos(-np.pi/3), -np.sin(-np.pi/3)],
                                   [np.sin(-np.pi/3), np.cos(-np.pi/3)]])
        
        vec = two_thirds - one_third
        rotated_vec = np.dot(rotation_matrix, vec)
        peak = one_third + rotated_vec
        
        new_points.extend([p0, one_third, peak, two_thirds])
    
    new_points.append(points[-1])
    return np.array(new_points)

def generate_koch_snowflake(iterations=4, scale=2.0, half_type='complete'):
    """Genera el copo de nieve completo o mitades"""
    # Generar copo completo primero
    steps = []
    current_points = koch_snowflake_external_step(None, 0, scale)
    steps.append(current_points)
    
    for i in range(1, iterations + 1):
        current_points = koch_snowflake_external_step(steps[-1], i, scale)
        steps.append(current_points)
    
    points_complete = steps[-1]
    
    # Filtrar puntos según el tipo de mitad
    if half_type == 'complete':
        return points_complete
    elif half_type == 'inferior':
        # Mitad inferior (y <= altura/2)
        mid_y = np.max(points_complete[:, 1]) / 2
        return points_complete[points_complete[:, 1] <= mid_y]
    elif half_type == 'superior':
        # Mitad superior (y >= altura/2)
        mid_y = np.max(points_complete[:, 1]) / 2
        return points_complete[points_complete[:, 1] >= mid_y]
    elif half_type == 'izquierda':
        # Mitad izquierda (x <= ancho/2)
        mid_x = np.max(points_complete[:, 0]) / 2
        return points_complete[points_complete[:, 0] <= mid_x]
    elif half_type == 'derecha':
        # Mitad derecha (x >= ancho/2)
        mid_x = np.max(points_complete[:, 0]) / 2
        return points_complete[points_complete[:, 0] >= mid_x]
    else:
        return points_complete

def create_koch_image(points, scale, iterations, color='blue', half_type='complete', filename=None):
    """Crea y guarda la imagen del copo de nieve"""
    # Crear figura sin mostrar ventana
    plt.ioff()  # Desactivar modo interactivo
    fig, ax = plt.subplots(figsize=(10, 10))
    
    try:
        # Dibujar el copo
        ax.plot(points[:, 0], points[:, 1], color, linewidth=1.5)
        
        # Solo rellenar si es completo
        if half_type == 'complete':
            ax.fill(points[:, 0], points[:, 1], 'lightblue', alpha=0.3)
        
        # Configuraciones
        ax.set_aspect('equal')
        ax.axis('off')
        
        title = f'Copo Koch'
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        
        # Ajustar límites según el tipo
        if half_type == 'complete':
            margin = 0.5 * scale
            ax.set_xlim(-margin, np.max(points[:, 0]) + margin)
            ax.set_ylim(-margin, np.max(points[:, 1]) + margin)
        elif half_type == 'inferior':
            ax.set_xlim(-0.5 * scale, np.max(points[:, 0]) + 0.5 * scale)
            ax.set_ylim(-0.1 * scale, np.max(points[:, 1]) + 0.1 * scale)
        elif half_type == 'superior':
            ax.set_xlim(-0.5 * scale, np.max(points[:, 0]) + 0.5 * scale)
            ax.set_ylim(np.min(points[:, 1]) - 0.1 * scale, np.max(points[:, 1]) + 0.1 * scale)
        elif half_type == 'izquierda':
            ax.set_xlim(-0.1 * scale, np.max(points[:, 0]) + 0.1 * scale)
            ax.set_ylim(-0.5 * scale, np.max(points[:, 1]) + 0.5 * scale)
        elif half_type == 'derecha':
            ax.set_xlim(np.min(points[:, 0]) - 0.1 * scale, np.max(points[:, 0]) + 0.1 * scale)
            ax.set_ylim(-0.5 * scale, np.max(points[:, 1]) + 0.5 * scale)
        
        # Guardar imagen si se proporciona filename
        if filename:
            plt.savefig(filename, dpi=150, bbox_inches='tight', facecolor='white', format='png')
            plt.close(fig)
            return filename
        else:
            # Convertir a base64 para mostrar en web
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight', facecolor='white')
            img_buffer.seek(0)
            img_data = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
            plt.close(fig)
            return img_data
            
    finally:
        # Asegurarse de cerrar la figura
        plt.close('all')
        plt.ion()  # Reactivar modo interactivo

@app.route('/', methods=['GET', 'POST'])
def index():
    """Página principal con formulario"""
    if request.method == 'POST':
        try:
            # Verificar si es una solicitud de borrado
            if 'clear_images' in request.form:
                clear_images()
                return redirect(url_for('index'))
            
            # Obtener parámetros del formulario
            level = int(request.form.get('level', 3))
            scale = float(request.form.get('scale', 2.0))
            color = request.form.get('color', 'blue')
            half_type = request.form.get('half_type', 'complete')
            
            # Validar parámetros
            if level < 0 or level > 8:
                return render_template('index.html', error="Las iteraciones deben estar entre 0 y 8")
            
            if scale <= 0 or scale > 10:
                return render_template('index.html', error="La escala debe estar entre 0.1 y 10")
            
            # Generar copo de nieve
            points = generate_koch_snowflake(level, scale, half_type)
            
            # Crear nombre de archivo único
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"koch_{half_type}_{level}iter_{scale}scale_{timestamp}.png"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            # Crear y guardar imagen
            create_koch_image(points, scale, level, color, half_type, filepath)
            
            # Obtener lista de imágenes existentes
            images = get_existing_images()
            
            return render_template('index.html', 
                                 level=level, 
                                 scale=scale, 
                                 color=color,
                                 half_type=half_type,
                                 images=images,
                                 outdir=app.config['UPLOAD_FOLDER'],
                                 success=True)
            
        except Exception as e:
            return render_template('index.html', error=f"Error: {str(e)}")
    
    # Método GET - mostrar formulario
    images = get_existing_images()
    return render_template('index.html', images=images, outdir=app.config['UPLOAD_FOLDER'])

def get_existing_images():
    """Obtener lista de imágenes existentes"""
    images = []
    if os.path.exists(app.config['UPLOAD_FOLDER']):
        for filename in sorted(os.listdir(app.config['UPLOAD_FOLDER']), reverse=True):
            if filename.endswith('.png'):
                images.append(f"/static/images/{filename}")
    return images

def clear_images():
    """Borrar todas las imágenes"""
    if os.path.exists(app.config['UPLOAD_FOLDER']):
        for filename in os.listdir(app.config['UPLOAD_FOLDER']):
            if filename.endswith('.png'):
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                try:
                    os.remove(filepath)
                except:
                    pass  # Ignorar errores al borrar

@app.route('/api/koch/generate', methods=['POST', 'GET'])
def generate_koch_api():
    """Endpoint API para generar copo de nieve"""
    try:
        # Obtener parámetros
        if request.method == 'POST':
            data = request.get_json() or {}
            iterations = int(data.get('iterations', 4))
            scale = float(data.get('scale', 2.0))
            color = data.get('color', 'blue')
            half_type = data.get('half_type', 'complete')
            return_image = data.get('return_image', True)
        else:
            iterations = int(request.args.get('iterations', 4))
            scale = float(request.args.get('scale', 2.0))
            color = request.args.get('color', 'blue')
            half_type = request.args.get('half_type', 'complete')
            return_image = request.args.get('return_image', 'true').lower() == 'true'
        
        # Validar parámetros
        if iterations < 0 or iterations > 8:
            return jsonify({'error': 'Iteraciones deben estar entre 0 y 8'}), 400
        if scale <= 0 or scale > 10:
            return jsonify({'error': 'Escala debe estar entre 0.1 y 10'}), 400
        if half_type not in ['complete', 'inferior', 'superior', 'izquierda', 'derecha']:
            return jsonify({'error': 'Tipo debe ser: complete, inferior, superior, izquierda, derecha'}), 400
        
        # Generar copo de nieve
        points = generate_koch_snowflake(iterations, scale, half_type)
        
        # Calcular métricas
        total_points = len(points)
        total_segments = total_points - 1
        estimated_length = total_segments * (scale / (3 ** iterations))
        
        # Preparar respuesta
        response_data = {
            'success': True,
            'metadata': {
                'iterations': iterations,
                'scale': scale,
                'color': color,
                'half_type': half_type,
                'total_points': total_points,
                'total_segments': total_segments,
                'estimated_length': round(estimated_length, 4),
                'generated_at': datetime.now().isoformat(),
                'fractal_dimension': round(np.log(4) / np.log(3), 4)
            }
        }
        
        # Si se solicita la imagen en la respuesta
        if return_image:
            image_base64 = create_koch_image(points, scale, iterations, color, half_type)
            response_data['image_base64'] = image_base64
        
        return jsonify(response_data)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/koch/list', methods=['GET'])
def list_images_api():
    """Endpoint API para listar imágenes"""
    try:
        images = []
        if os.path.exists(app.config['UPLOAD_FOLDER']):
            for filename in sorted(os.listdir(app.config['UPLOAD_FOLDER']), reverse=True):
                if filename.endswith('.png'):
                    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    file_info = {
                        'filename': filename,
                        'url': f"/static/images/{filename}",
                        'size': os.path.getsize(filepath),
                        'created': datetime.fromtimestamp(os.path.getctime(filepath)).isoformat()
                    }
                    images.append(file_info)
        
        return jsonify({'images': images, 'total': len(images)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/static/images/<filename>')
def serve_image(filename):
    """Servir imágenes estáticas"""
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if os.path.exists(filepath):
        return send_file(filepath)
    else:
        return "Imagen no encontrada", 404

@app.route('/api/koch/clear', methods=['POST'])
def clear_images_api():
    """Endpoint API para borrar imágenes"""
    try:
        clear_images()
        return jsonify({'success': True, 'message': 'Todas las imágenes han sido eliminadas'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.errorhandler(404)
def not_found(error):
    return render_template('index.html', error="Página no encontrada"), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('index.html', error="Error interno del servidor"), 500

if __name__ == '__main__':
    print("🚀 Iniciando Koch Snowflake Generator...")
    print("🌐 Servidor web disponible en: http://localhost:5000")
    print("🔧 Backend: Matplotlib con Agg (sin GUI)")
    print("✅ Error de Tkinter solucionado")
    
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)