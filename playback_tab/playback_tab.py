import panel as pn

import os
import panel as pn

DEFAULT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'experiments')

class PlaybackTab:
    def __init__(self):
        self.dir_select = pn.widgets.TextInput(name='Pasta de vídeos', value=DEFAULT_DIR, width=400)
        self.refresh_button = pn.widgets.Button(name='🔄 Atualizar', button_type='primary', width=100)
        self.video_select = pn.widgets.Select(name='Vídeos disponíveis', options=self._list_videos(DEFAULT_DIR), width=400)
        self.video_pane = pn.pane.HTML(self._empty_video_html(), width=650, height=400)
        self.status_text = pn.pane.Markdown('Selecione um vídeo para exibir.')
        self.refresh_button.on_click(self._refresh)
        self.video_select.param.watch(self._update_video, 'value')
        self.dir_select.param.watch(self._refresh, 'value')

    def _empty_video_html(self):
        return "<div style='width:640px; height:360px; border:2px dashed #ccc; display:flex; align-items:center; justify-content:center; color:#888;'>Nenhum vídeo selecionado</div>"

    def _list_videos(self, folder):
        if not os.path.isdir(folder):
            return {'Nenhum vídeo encontrado': ''}
        files = [f for f in os.listdir(folder) if f.lower().endswith('.mp4')]
        if not files:
            return {'Nenhum vídeo encontrado': ''}
        return {f: os.path.join(folder, f) for f in sorted(files, reverse=True)}

    def _refresh(self, *events):
        folder = self.dir_select.value
        self.video_select.options = self._list_videos(folder)
        self.video_select.value = ''
        self.video_pane.object = self._empty_video_html()
        self.status_text.object = 'Selecione um vídeo para exibir.'

    def _update_video(self, event):
        video_path = event.new
        if not video_path or not os.path.isfile(video_path):
            self.video_pane.object = self._empty_video_html()
            self.status_text.object = 'Selecione um vídeo para exibir.'
            return
        # Usar iframe para melhor compatibilidade com vídeos mp4
        video_filename = os.path.basename(video_path)
        video_url = f'/experiments/{video_filename}'
        video_html = f"""
        <iframe src='{video_url}' width='640' height='360' frameborder='0' allowfullscreen>
            <p>Seu navegador não suporta iframes.</p>
            <p>Acesse diretamente: <a href='{video_url}' target='_blank'>{video_filename}</a></p>
        </iframe>
        <br>
        <p><a href='{video_url}' target='_blank'>🔗 Abrir vídeo em nova aba</a></p>
        """
        self.video_pane.object = video_html
        self.status_text.object = f'Exibindo: {video_filename}'


    def get_panel(self):
        return pn.Column(
            pn.pane.Markdown('## Playback'),
            pn.Row(self.dir_select, self.refresh_button),
            pn.Spacer(height=10),
            self.video_select,
            pn.Spacer(height=10),
            self.video_pane,
            pn.Spacer(height=10),
            self.status_text,
            margin=(10, 0)
        )

def get_tab():
    return PlaybackTab().get_panel()

